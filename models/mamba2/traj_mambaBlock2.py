# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from .ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm import Mamba2

# GPS_SSM + Route_SSM
class TrajMamba2(nn.Module):
    def __init__(
        self,
        d_model, 
        d_inner=0, 
        d_state=128, 
        d_conv=4, 
        conv_init=None,
        expand=2, 
        headdim=64, 
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1, 
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True, 
        norm_before_gate=False, 
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False, 
        conv_bias=True, 
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True, 
        device=None,
        dtype=None,
        aux_feature_size=0, 
        use_S4_bcdt=False, # 是否能使用S4形式（而非S6）构造B, C, dt
        add_bcdt_item = True, 
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = d_inner // self.world_size if d_inner else (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == d_inner if d_inner else self.expand * self.d_model 
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        self.use_S4_bcdt = use_S4_bcdt
        if aux_feature_size or (aux_feature_size==0 and self.use_S4_bcdt):
            self.no_gen_bcdt = True 
        else: self.no_gen_bcdt = False
        self.add_bcdt_item = add_bcdt_item
        
        self.B, self.C, self.dt, self.R_B, self.R_C, self.R_dt = None, None, None, None, None, None
        if self.no_gen_bcdt and (self.use_S4_bcdt or self.add_bcdt_item):   
            init_BC = torch.randn(4, self.ngroups * self.d_state, dtype=torch.float32, device=device)
            init_dt = torch.randn(2, self.nheads, dtype=torch.float32, device=device)      
            
            self.B = nn.Parameter(init_BC[0])
            self.B._no_weight_decay = True    
            self.C = nn.Parameter(init_BC[1])
            self.C._no_weight_decay = True
            self.dt = nn.Parameter(init_dt[0]) 
            self.dt._no_weight_decay = True
            
            # used in no share item version
            self.R_B = nn.Parameter(init_BC[2])
            self.R_B._no_weight_decay = True    
            self.R_C = nn.Parameter(init_BC[3])
            self.R_C._no_weight_decay = True
            self.R_dt = nn.Parameter(init_dt[1])
            self.R_dt._no_weight_decay = True


        # Order: [z, x, (B, C, dt)]   z,x: self.d_inner;  B,C: self.ngroups * self.d_state; dt: self.nheads
        d_in_proj = 2 * self.d_inner if self.no_gen_bcdt else 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm if self.no_gen_bcdt else self.d_ssm + 2 * self.ngroups * self.d_state 
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim, 
            padding=d_conv - 1,
            **factory_kwargs,
        ) # B*in_channels*L → B*out_channels*(L + d_conv-1)     in_channels=out_channels=conv_dim
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU() 

        '''GPS-SSM'''
        # Initialize log dt bias    so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range) # (nheads)
        A_log = torch.log(A).to(dtype=dtype) # also Keep A_log in fp32 in update version: delete ".to(dtype=dtype)"
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True


        '''Route-SSM'''
        self.R_bcdt_proj = nn.Linear(self.d_ssm, 2 * self.ngroups * self.d_state + self.nheads, bias=bias, **factory_kwargs)
        
        # Initialize log dt bias    so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.R_dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.R_dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range) # (nheads)
        A_log = torch.log(A).to(dtype=dtype) # also Keep A_log in fp32 in update version: delete ".to(dtype=dtype)"
        self.R_A_log = nn.Parameter(A_log)
        self.R_A_log._no_weight_decay = True

        # D "skip" parameter
        self.R_D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.R_D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate, # False√
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model//2, bias=bias, **factory_kwargs)
            self.R_out_proj = nn.Linear(self.d_inner, self.d_model//2, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model//2, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)
            self.R_out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model//2, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, B=None, C=None, dt=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        assert not (B is None and self.no_gen_bcdt and not self.use_S4_bcdt) # 此时无法获取任何形式的B,C,dt
        assert not (B is not None and not self.no_gen_bcdt) # 此时模型被定义为由输入构造ssm参数的模式，不会使用额外传入的B,C,dt

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)

        seqlen_og = seqlen
        if seqlen is None: 
            batch, seqlen, dim = u.shape
        else: 
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state) 
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit) # ？
        
        if B is not None and self.add_bcdt_item:
            B = B + self.B
            C = C + self.C
            dt = dt + self.dt
        elif B is None and self.no_gen_bcdt:
            B = repeat(self.B, "N -> b l N", b=batch, l=seqlen).to(dtype=torch.float32)
            C = repeat(self.C, "N -> b l N", b=batch, l=seqlen).to(dtype=torch.float32)
            dt = repeat(self.dt, "h -> b l h", b=batch, l=seqlen).to(dtype=torch.float32)
        

        """ d_mlp = self.d_inner - self.d_ssm """
        if self.no_gen_bcdt:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm) // 2
            z0, x0, z, xBC = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm],
                dim=-1
            )
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )

        assert self.activation in ["silu", "swish"]
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)] # b l d -> b d l -> b l d
            )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2)
        
        if self.no_gen_bcdt: 
            x = xBC
        else:
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim), # (B, L, self.nheads, self.headdim)
            dt, # (B, L, self.nheads)
            A, # (nheads)
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups), # (B, L, self.ngroups, self.d_state)
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups), # (B, L, self.ngroups, self.d_state)
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D, # (self.nheads, self.headdim) / (self.nheads,)
            z=None, 
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            **dt_limit_kwargs,
            return_final_states=False,
            return_varlen_states=False 
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Route-SSM
        R_x = self.act(z)
        R_A = -torch.exp(self.R_A_log.float())  # (nheads) or (d_inner, d_state) 
        
        R_bcdt = self.R_bcdt_proj(y)
        R_B, R_C, R_dt = torch.split(R_bcdt, [self.ngroups * self.d_state, self.ngroups * self.d_state, self.nheads], dim=-1)
        if self.add_bcdt_item:
            R_B = R_B + self.B
            R_C = R_C + self.C
            R_dt = R_dt + self.dt
        
        R_y = mamba_chunk_scan_combined(
            rearrange(R_x, "b l (h p) -> b l h p", p=self.headdim), # (B, L, self.nheads, self.headdim)
            R_dt, # (B, L, self.nheads)
            R_A, # (nheads)
            rearrange(R_B, "b l (g n) -> b l g n", g=self.ngroups), # (B, L, self.ngroups, self.d_state)
            rearrange(R_C, "b l (g n) -> b l g n", g=self.ngroups), # (B, L, self.ngroups, self.d_state)
            chunk_size=self.chunk_size,
            D=rearrange(self.R_D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.R_D, # (self.nheads, self.headdim) / (self.nheads,)
            z=None,
            dt_bias=self.R_dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            **dt_limit_kwargs,
            return_final_states=False, 
            return_varlen_states=False 
        )
        R_y = rearrange(R_y, "b l h p -> b l (h p)")

        if self.rmsnorm:
            y = self.norm(y, z) 
        else:
            y = y * F.silu(z)
        
        if d_mlp > 0: 
            y = torch.cat([F.silu(z0) * x0, y], dim=-1) # (B, L, d_ssm) -> (B, L, d_inner)
            R_y = torch.cat([F.silu(z0) * x0, R_y], dim=-1) # (B, L, d_ssm) -> (B, L, d_inner)
        
        if seqlen_og is not None: 
            y = rearrange(y, "b l d -> (b l) d")
            R_y = rearrange(R_y, "b l d -> (b l) d")
        
        out = self.out_proj(y) # (B, L, d_inner) -> (B, L, D)
        R_out = self.R_out_proj(R_y)
        
        return torch.cat([R_out, out], dim=-1) # z, x



class Mamba2_fix(Mamba2): 
    # mamba_ssm包中，ssd_combined.py文件中的函数mamba_split_conv1d_scan_combined存在Bug，修正
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            conv_init=conv_init,
            expand=expand,
            headdim=headdim,
            d_ssm=d_ssm,
            ngroups=ngroups,
            A_init_range=A_init_range,
            D_has_hdim=D_has_hdim,
            rmsnorm=rmsnorm,
            norm_before_gate=norm_before_gate,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            dt_limit=dt_limit,
            bias=bias,
            conv_bias=conv_bias,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            layer_idx=layer_idx,
            process_group=process_group,
            sequence_parallel=sequence_parallel,
            device=device,
            dtype=dtype
        )
    
    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=False,
                return_varlen_states=cu_seqlens is not None,
            )
            
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out