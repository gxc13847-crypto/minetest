import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat,rearrange
from mamba_ssm import Mamba

from .mamba2.mamba2 import TrajMixerModel2
from .encode import PositionalEmbedding, FourierEncode
from data import MAX_TRIP_LEN, COL_I
from utils import get_batch_mask, tokenize_timestamp, pack_input, pad_input, lamda_scheduler

class GateMask(nn.Module):
    def __init__(self, 
                    d_model,
                    expand: int = 2,
                    sigma: float = 0.5,
                    pooling_method: str = 'sigmoid',
                    device=None,
                    ):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(MAX_TRIP_LEN, d_model))
        self.sigma = sigma
        self.based = 0.5
        self.pooling_method = pooling_method
        self.device = device
        self.trend_func = Mamba(d_model=d_model, expand=expand, device=device, dtype=torch.float32)
    
        # 归一化, 单独用Mamba缺少Block的LayerNorm（避免中间值漂移）
        self.pre_norm = torch.nn.LayerNorm(d_model, eps=1e-5, device=self.device)
        self.final_norm = torch.nn.LayerNorm(d_model, eps=1e-5, device=self.device)
    
    def forward(self, hidden_state, valid_lens, batch_arr, batch_mask, cu_seqlens=None, seq_idx=None):
        """        
        :param hidden_states: Shape is [B,L,H]
        :param batch_mask: Shape is [B,L]
        :param seq_idx: 
        :param cu_seqlens: 
        :return: 
            input_mask: Shape is [B,L]
            new_valid_lens
            new_max_length
        """
        seq_len = hidden_state.size(1)
        mu = self.mu
        if seq_len > mu.size(0):
            padding = torch.zeros(
                seq_len - mu.size(0),
                mu.size(1),
                dtype=mu.dtype,
                device=mu.device,
            )
            mu = torch.cat([mu, padding], dim=0)
        mu = mu[:seq_len].unsqueeze(0)
        
        mu_ = self.refactor_mask(mu, hidden_state, seq_idx, cu_seqlens)  # (B, L, H)
        mu_ = mu_.topk(k=3, dim=-1)[0].mean(dim=-1) # (B, L, H) -> (B, L) 以token为单位进行mask
        
        mask = mu_ if not self.training else \
                                    mu_ + torch.normal(0, self.sigma, mu_.shape, device=self.device)
        input_mask = self.hard_sigmoid(mask.masked_fill(batch_mask, 0))
        # 确保起点、终点不被mask
        input_mask[:,0] = 1.0
        input_mask[batch_arr, valid_lens-1] = 1.0
        
        new_valid_lens = (input_mask!=0).sum(dim=-1) # 计算mask后的每条数据的有效长度
        new_L = new_valid_lens.max().item() # 计算新的最大长度
        
        return input_mask, new_valid_lens, new_L, mu_

    def refactor_mask(self, mu, x, seq_idx=None, cu_seqlens=None):
        """        
        :param mu: Shape is [L,1]
        :param x: Shape is [L,H]
        :param seq_idx: 
        :param cu_seqlens: 
        :return: mu: Shape is [L,H]
        """
        # trend = self.trend_func(x)
        residual = x
        output = self.pre_norm(residual)
        output = self.trend_func(output)
        trend = self.final_norm(output+residual) # 残差+归一化

        mu = mu + self.based

        mu = mu * torch.sigmoid(mu * trend)
        
        return mu
    
    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

 
class MEC(nn.Module):
    """
    Maximum Entropy Coding loss for contrastive between two views.
    Liu X, Wang Z, Li Y, et al. Self-Supervised Learning via Maximum Entropy Coding. NeuralIPS 2022.
    """

    def __init__(self, embed_dim, num_epoch, num_iter, batch_size, 
                 hidden_size=256, n=5, eps=512, warmup_epoch=10): #, teachers
        super().__init__()

        # The predictor is applied on top of the encoder as a non-linear casting.
        self.predictor = nn.Sequential(nn.Linear(embed_dim, hidden_size, bias=False),
                                       nn.BatchNorm1d(hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_size, embed_dim))

        # # The teachers can be symmetric or asymmetric models of the encoders.
        # self.teachers = nn.ModuleList(teachers)

        self.n = n # 5
        lamda = 1 / (batch_size * eps / embed_dim) # d/Nϵ2
        self.lamda_schedule = lamda_scheduler(8/lamda, 1/lamda, num_epoch, num_iter,
                                                    warmup_epochs=warmup_epoch)

    def _predict_single(self, z):
        was_training = self.predictor.training
        self.predictor.eval()
        z = self.predictor(z)
        if was_training:
            self.predictor.train()
        return z

    def forward(self, z, p, batch_size, it):
        # z1, z2 = (encoder(trip, valid_len) for encoder in encoders)
        # z1, z2 = self.predictor(z1), self.predictor(z2)
        # with torch.no_grad():
        #     # The teachers are totally detached from the gradient updating.
        #     # Instead, they can be updated using the momentum trainer.
        #     p1, p2 = (teacher(trip, valid_len) for teacher in self.teachers)
        # p1 = p1.detach()
        # p2 = p2.detach()
        z = self.predictor(z) if z.size(0) > 1 else self._predict_single(z) if z.size(0) > 1 else self._predict_single(z)

        # Symmetric loss between two views.
        lamda_inv = self.lamda_schedule[it]
        loss = self.mec(p, z, lamda_inv) / batch_size
        # loss = (self.mec(p1, z2, lamda_inv) + self.mec(p2, z1, lamda_inv)) * 0.5 / trip.shape[0]
        loss = -1 * loss * lamda_inv
        return loss

    def mec(self, view1, view2, lamda_inv):
        view1, view2 = F.normalize(view1), F.normalize(view2)
        c = torch.mm(view1, view2.transpose(0, 1)) / lamda_inv  # (B, B)
        power = c
        sum_p = torch.zeros_like(power)
        for k in range(1, self.n+1):
            if k > 1:
                power = torch.mm(power, c)
            if (k + 1) % 2 == 0:
                sum_p += power / k
            else:
                sum_p -= power / k
        trace = torch.trace(sum_p)
        return trace


class Trajectory_Mamba(nn.Module):
    def __init__(self, embed_size, d_model, num_road, 
                spatial_border, high_order_feature_border, temporal_border,
                device, use_higher_features=True, n_layer=4, d_state=128, headdim=64, d_inner=0, 
                use_S4_form=False, add_bcdt_item=False, mask_weight=0.5, 
                ):
        """
        Args:
            embed_size (int): dimension of learnable embedding modules.
            d_model (int): dimension of the sequential models.
            num_road (int): number of roads.
            spatial_border (list): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
            high_order_feature_border (list): coordinates indicating the high order feature border: [[x_min, y_min], [x_max, y_max]].
            use_higher_features (bool, optional): whether to use trajectory's higher-order features. Defaults to True.
            n_layer (int, optional): number of stacked Traj-Mamba Blocks. Defaults to 4.
            d_state (int, optional): state size of Traj-SSM in Traj-Mamba Block. Defaults to 128.
            headdim (int, optional): head dimension of Traj-SSM in Traj-Mamba Block. Defaults to 64.
            d_inner (int, optional): inner model dimension of Traj-Mamba Blocks. If setting to 0 means d_inner=2*d_model.
            mask_weight (float, optional): loss weight of mask loss. Defaults to 0.5.
        """

        super().__init__()
        assert d_model % 2 == 0
        self.output_size = d_model
        self.num_road = num_road
        self.spatial_size = len(COL_I["spatial"])
        self.temporal_size = len(COL_I['temporal'])
        self.high_order_feature_size = len(COL_I['high_order_features'])
        self.device = device
        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)
        self.temporal_border = nn.Parameter(torch.tensor(temporal_border), requires_grad=False)
        self.high_order_feature_border = nn.Parameter(torch.tensor(high_order_feature_border), requires_grad=False)
        self.use_higher_features = use_higher_features

        self.mask_generator = GateMask(d_model=self.spatial_size+self.temporal_size+1, device=device) 

        ST_embed_size = d_model // 2
        self.traj_view = nn.ModuleDict({
            'spatial_embed_layer': nn.Sequential(nn.Linear(len(COL_I["spatial"]), embed_size), nn.LeakyReLU(), nn.Linear(embed_size, ST_embed_size)),
            'temporal_embed_modules': nn.ModuleList([FourierEncode(embed_size) for _ in range(5)]),
            'fine_temporal_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 2, ST_embed_size)), # 细粒度时间特征
            'coarse_temporal_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 3, ST_embed_size)), # 粗粒度时间特征
            'road_index_embed_layer': nn.Sequential(nn.Embedding(num_road, embed_size),
                                               nn.LayerNorm(embed_size),
                                               nn.Linear(embed_size, ST_embed_size)),
            'seq_encoder': TrajMixerModel2(d_model=d_model, n_layer=n_layer, d_intermediate=0, 
                            aux_feature_size=self.high_order_feature_size if self.use_higher_features else 0, 
                            d_state=d_state, headdim=headdim, d_inner=d_inner, use_S4_form=use_S4_form, add_bcdt_item=add_bcdt_item, 
                            device=device, dtype=torch.float32) 
        })
        
        self.mask_weight = mask_weight
    
    def forward(self, input_seq, valid_lens):
        """
        Args:
            input_seq (FloatTensor): batch of trajectory features, with shape (B, L, F).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.

        Returns:
            Tensor: embedding vectors for this batch of trajectories, with shape (B, E).
        """
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        norm_temporal = (temporal - self.temporal_border[0].unsqueeze(0).unsqueeze(0)) / \
                            (self.temporal_border[1] - self.temporal_border[0]).unsqueeze(0).unsqueeze(0)
        norm_road = road / (self.num_road -1)
        mask_generator_input = torch.cat([norm_spatial, norm_temporal, norm_road.unsqueeze(-1)],dim=-1) 
        high_order_features = input_seq[:, :, COL_I['high_order_features']] # (B, L, 3)
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        traj_h, new_valid_lens, _  = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, mask_generator_input)

        return traj_h
    

    def cal_traj_h(self, norm_spatial, temporal, road, aux_features, valid_lens, mask_generator_input=None):
        """Calculate trajectories' embedding vectors given their spatio-temporal features.
        The detailed definition of trajectories' spatial and temporal features can be refered to `data.py`.

        Args:
            norm_patial (FloatTensor): trajectories' spatial features, with shape (B, L, F_s).
            temporal (FloatTensor): trajectories' temporal features, with shape (B, L, F_t).
            road (LongTensor): trajectories' road index, with shape (B, L).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.
            aux_features (FloatTensor): trajectories' high-order features, with shape (B, L, F_h) or None.

        Returns:
            FloatTensor: the embedding vectors of this batch of trajectories, with shape (B, E).
        """
        B, L = norm_spatial.size(0), norm_spatial.size(1)
        batch_mask = get_batch_mask(B, L, valid_lens)
        batch_arr = torch.arange(end=B,dtype=torch.int32,device=valid_lens.device)
        
        mask_generator_input = mask_generator_input.masked_fill(batch_mask.unsqueeze(-1), 0.0)
        input_mask, new_valid_lens, new_L, mu_ = self.mask_generator(mask_generator_input, valid_lens, batch_arr, batch_mask)
        
        new_batch_mask = get_batch_mask(B, new_L, new_valid_lens)

        spatial_e = self.traj_view['spatial_embed_layer'](norm_spatial)  # norm_spatial (B, L, E) 

        temporal_tokens = [self.traj_view['temporal_embed_modules'][i](temp_token)
                        for i, temp_token in enumerate(tokenize_timestamp(temporal))]
        fine_temporal_e = self.traj_view['fine_temporal_embed_layer'](torch.cat(temporal_tokens[-2:], -1))
        coarse_temporal_e = self.traj_view['coarse_temporal_embed_layer'](torch.cat(temporal_tokens[:-2], -1))
        # Temporal values lower than 0 stands for feature mask.
        temporal_mask = temporal[..., :1] < 0
        fine_temporal_e = fine_temporal_e.masked_fill(temporal_mask, 0)
        coarse_temporal_e = coarse_temporal_e.masked_fill(temporal_mask, 0)

        road_e = self.traj_view['road_index_embed_layer'](road) 

        traj_h = torch.cat([road_e + coarse_temporal_e, spatial_e + fine_temporal_e], dim=-1)
        

        masked_traj_h, indices = pack_input(traj_h, input_mask)
        masked_aux_features = rearrange(aux_features, "b s ... -> (b s) ...")[indices]
        
        new_seq_idx, new_indices = pack_input(repeat(batch_arr, "B -> B L", L=new_L), ~new_batch_mask)
        new_cu_seqlens = F.pad(torch.cumsum(new_valid_lens, dim=0, dtype=torch.int32), (1, 0))
        # assert len(masked_traj_h)==new_cu_seqlens[-1]
        masked_traj_h = self.traj_view['seq_encoder'](masked_traj_h, masked_aux_features.unsqueeze(0), 
                                                      new_cu_seqlens[-1], new_seq_idx.unsqueeze(0), new_cu_seqlens)
        traj_h = pad_input(masked_traj_h, new_indices, B, new_L).sum(1) / new_valid_lens.unsqueeze(-1)

        return traj_h, new_valid_lens, mu_ 

    
    def loss(self, input_seq, valid_lens, teacher_traj_h, MEC_weight=0, MEC_loss=None, it=None):
        """
        Args:
            Same as the forward function.

        Returns:
            FloatTensor: the pre-training loss value of this batch.
        """
        s_time = time.time()
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        norm_temporal = (temporal - self.temporal_border[0].unsqueeze(0).unsqueeze(0)) / \
                            (self.temporal_border[1] - self.temporal_border[0]).unsqueeze(0).unsqueeze(0)
        norm_road = road / (self.num_road -1)
        mask_generator_input = torch.cat([norm_spatial, norm_temporal, norm_road.unsqueeze(-1)],dim=-1) 
        high_order_features = input_seq[:, :, COL_I['high_order_features']]
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        e_time = time.time()
        
        traj_h, new_valid_lens, mu_ = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, mask_generator_input)
        
        mask_loss = self.cal_mask_loss(mu_, valid_lens)
        MMTEC_loss = MEC_loss(traj_h, teacher_traj_h, traj_h.size(0), it) if MEC_weight else 0

        loss = self.mask_weight * mask_loss + MEC_weight * MMTEC_loss
        return loss, new_valid_lens, (e_time-s_time)
    
    def cal_mask_loss(self, mu_, valid_lens):
        if len(mu_.shape) > 1:
            batch_mask = get_batch_mask(mu_.size(0), mu_.size(1), valid_lens)
            mu_ = mu_.masked_fill(batch_mask, 0)
        reg = 0.5 + 0.5 * torch.erf(mu_ / (self.mask_generator.sigma * np.sqrt(2)))
        return reg.mean()
    
    def forward_on_search_mode(self, input_seq, valid_lens):
        '''Calculate trajectories' embedding vectors given their spatio-temporal features and get embedding time of model.
        '''
        s_time = time.time()
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        norm_temporal = (temporal - self.temporal_border[0].unsqueeze(0).unsqueeze(0)) / \
                            (self.temporal_border[1] - self.temporal_border[0]).unsqueeze(0).unsqueeze(0)
        norm_road = road / (self.num_road -1)
        mask_generator_input = torch.cat([norm_spatial, norm_temporal, norm_road.unsqueeze(-1)],dim=-1) 
        high_order_features = input_seq[:, :, COL_I['high_order_features']]
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        e_time = time.time()
        
        start_time = time.time()
        traj_h, new_valid_lens, _ = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, mask_generator_input)
        end_time = time.time()

        return traj_h, (end_time - start_time), (e_time-s_time) 
    
    def cal_compressed_traj(self, input_seq, valid_lens):
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        norm_temporal = (temporal - self.temporal_border[0].unsqueeze(0).unsqueeze(0)) / \
                            (self.temporal_border[1] - self.temporal_border[0]).unsqueeze(0).unsqueeze(0)
        # norm_spatial = input_seq[:, :, COL_I['norm_spatial']] # 修改，不知道原先的那种归一化方法（pad值归一化后巨大）是不是导致mask_generator输出mu_有问题的原因。。。
        # norm_temporal = input_seq[:, :, COL_I['norm_temporal']]
        norm_road = road / (self.num_road -1)
        mask_generator_input = torch.cat([norm_spatial, norm_temporal, norm_road.unsqueeze(-1)],dim=-1) 
        # aux_features = self.cal_high_order_features(spatial, temporal, valid_lens)
        # high_order_features = input_seq[:, :, COL_I['high_order_features']] # (B, L, 3)
        # norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
        #     (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        # aux_features = torch.cat([norm_high, road.unsqueeze(-1)], dim=-1)
        B, L = norm_spatial.size(0), norm_spatial.size(1)
        # positions = repeat(torch.arange(L), 'L -> B L', B=B).to(valid_lens.device)
        # pos_encoding = self.pos_encode_layer(positions) # (B,L,D)
        batch_mask = get_batch_mask(B, L, valid_lens)
        batch_arr = torch.arange(end=B,dtype=torch.int32,device=valid_lens.device)
        # seq_idx = torch.cat([torch.full((valid_len,), i, dtype=torch.int32, device=valid_lens.device)
        #                     for i, valid_len in enumerate(valid_lens)], dim=0).unsqueeze(0)
        # cu_seqlens = F.pad(torch.cumsum(valid_lens, dim=0, dtype=torch.int32), (1, 0))
        mask_generator_input = mask_generator_input.masked_fill(batch_mask.unsqueeze(-1), 0.0)
        input_mask, new_valid_lens, new_L, mu_ = self.mask_generator(mask_generator_input, valid_lens, batch_arr, batch_mask) #, cu_seqlens
        
        new_batch_mask = get_batch_mask(B, new_L, new_valid_lens)
        new_seq_idx, new_indices = pack_input(repeat(batch_arr, "B -> B L", L=new_L), ~new_batch_mask)
        masked_input_seq, indices = pack_input(input_seq, input_mask)
        masked_input_seq = pad_input(masked_input_seq, new_indices, B, new_L)
        return masked_input_seq, new_valid_lens, new_L