# pretrain framework
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import repeat, rearrange
import time

from .encode import PositionalEmbedding, FourierEncode
from data import COL_I, pad_batch
from utils import cal_tensor_geo_distance, get_batch_mask, tokenize_timestamp, pack_input, pad_input
from .mamba2.mamba2 import TrajMixerModel2


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, neighbors_mask, batch_mask):
        N = encoder_outputs.shape[2]
        hidden = hidden.unsqueeze(2).repeat(1, 1, N, 1) # (B,L,D) -> (B,L,N,D)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1))) # (B,L,N,D)

        attention = self.v(energy).squeeze(-1) # (B,L,N)
        attention = attention.masked_fill(batch_mask.unsqueeze(-1) | neighbors_mask, -1e10)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=-1) # (B,L,N)



class TrajClip(nn.Module):
    def __init__(self, embed_size, d_model, 
                 road_neighbors,
                 spatial_border, high_order_feature_border, device,
                 road_embed=None, num_road=None,
                 poi_embed=None, poi_coors=None, 
                 use_higher_features=True, n_layer=4, d_state=128, headdim=64, d_inner=0, use_S4_form=False, add_bcdt_item=False,
                 road_weight=1, poi_weight=1, pois_dist_thres=300, poi_max_neighbors=10, 
                 road_transW_lambda=1.0, poi_distW_lambda=0.5,
                 use_poi_view=True, use_textual_road_embed=True, use_road_in_traj_view=True,
                 **kwargs
                 ):
        """The core model of Trajectory CLIP.

        Args:
            embed_size (int): dimension of learnable embedding modules.
            d_model (int): dimension of the sequential models.
            road_embed (np.array): pre-defined embedding matrix of roads, with shape (n_roads, E).
            poi_embed (np.array): pre-defined embedding matrix of POIs, with shape (n_pois, E).
            poi_coors (np.array): coordiantes of all POIs, with shape (n_pois, 2).
            spatial_border (list): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
            road_weight (int, optional): loss weight of road view. Defaults to 1.
            poi_weight (int, optional): loss weight of poi view. Defaults to 1.
            use_higher_features (bool, optional): whether to use trajectory's higher-order features. Defaults to True.
            n_layer (int, optional): number of stacked Traj-Mamba Blocks. Defaults to 4.
            d_state (int, optional): state size of Traj-SSM in Traj-Mamba Block. Defaults to 128.
            headdim (int, optional): head dimension of Traj-SSM in Traj-Mamba Block. Defaults to 64.
            d_inner (int, optional): inner model dimension of Traj-Mamba Blocks. If setting to 0 means d_inner=2*d_model.
        """

        super().__init__()
        assert d_model % 2 == 0
        if num_road is None:
            if road_embed is not None:
                num_road = road_embed.shape[0]
            else:
                num_road = road_neighbors["road_neighbors_sets"].shape[0]
        self.num_road = int(num_road)
        self.use_textual_road_embed = bool(use_textual_road_embed and road_embed is not None)
        self.use_road_in_traj_view = bool(use_road_in_traj_view)

        self.road_neighbors = nn.Parameter(torch.from_numpy(road_neighbors["road_neighbors_sets"]), requires_grad=False)
        self.road_neighbors_num = nn.Parameter(torch.from_numpy(road_neighbors["road_neighbors_num"]), requires_grad=False)
        self.road_neighbors_weight = nn.Parameter(torch.from_numpy(road_neighbors["road_neighbors_weight"]).float(), requires_grad=False) # length weight, transfer weight

        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)
        self.high_order_feature_border = nn.Parameter(torch.tensor(high_order_feature_border), requires_grad=False)
        
        self.road_weight = road_weight
        self.poi_weight = poi_weight
        self.use_poi_view = use_poi_view
        self.use_higher_features = use_higher_features

        ST_embed_size = d_model // 2
        self.traj_view = nn.ModuleDict({
            'spatial_embed_layer': nn.Sequential(nn.Linear(len(COL_I["spatial"]), embed_size), nn.LeakyReLU(), nn.Linear(embed_size, ST_embed_size)),
            'temporal_embed_modules': nn.ModuleList([FourierEncode(embed_size) for _ in range(5)]),
            'fine_temporal_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 2, ST_embed_size)), # 细粒度时间特征
            'coarse_temporal_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 3, ST_embed_size)), # 粗粒度时间特征
            'road_index_embed_layer': nn.Sequential(nn.Embedding(self.num_road, embed_size),
                                               nn.LayerNorm(embed_size),
                                               nn.Linear(embed_size, ST_embed_size)),
            'seq_encoder': TrajMixerModel2(d_model=d_model, n_layer=n_layer, d_intermediate=0, 
                            aux_feature_size=len(COL_I['high_order_features']) if self.use_higher_features else 0, 
                            d_state=d_state, headdim=headdim, d_inner=d_inner, use_S4_form=use_S4_form, add_bcdt_item=add_bcdt_item, 
                            device=device, dtype=torch.float32) 
        })

        self.road_transW_lambda = road_transW_lambda
        self.road_view = nn.ModuleDict({
            'road_neighbors_att': Attention(d_model),
            'neighbors_weight_layer': nn.Linear(d_model, d_model, bias=False),
            'OD_weight_layer': nn.Linear(d_model, d_model, bias=False),
            'order_embed_module': FourierEncode(embed_size),
            'order_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size, d_model)),
            'time_embed_module': FourierEncode(embed_size),
            'time_embed_layer': nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size, d_model)),
            'merge_norm': nn.Sequential(nn.BatchNorm1d(d_model),
                                        nn.ReLU(inplace=True)), 
            'seq_encoder': TrajMixerModel2(d_model=d_model, n_layer=2, d_intermediate=0, aux_feature_size=0, 
                            d_inner=d_inner if d_model<=d_inner else d_model, # TODO
                            block_type="Mamba2", device=device, dtype=torch.float32) 
        })
        if self.use_textual_road_embed:
            road_embed_mat = nn.Embedding(*road_embed.shape) 
            road_embed_mat.weight = nn.Parameter(torch.from_numpy(road_embed).float(), requires_grad=False)
            self.road_view.update({
                'text_embed_mat': road_embed_mat,
                'text_embed_layer': nn.Sequential(nn.LayerNorm(road_embed.shape[1]),
                                                  nn.Linear(road_embed.shape[1], d_model)),
            })
        else:
            self.road_view.update({
                'road_id_embed_layer': nn.Sequential(nn.Embedding(self.num_road, embed_size),
                                                     nn.LayerNorm(embed_size),
                                                     nn.Linear(embed_size, d_model)),
            })

        if self.use_poi_view:
            if poi_embed is None or poi_coors is None:
                raise ValueError("poi_embed and poi_coors are required when use_poi_view=True")
            self.poi_coors = nn.Parameter(torch.from_numpy(poi_coors).float(), requires_grad=False) # n_pois：12439[chengdu]
            self.pois_dist_thres = pois_dist_thres
            self.poi_max_neighbors = poi_max_neighbors
            self.construct_poi_neighbors()

            self.poi_distW_lambda = poi_distW_lambda
            poi_embed_mat = nn.Embedding(*poi_embed.shape) 
            poi_embed_mat.weight = nn.Parameter(torch.from_numpy(poi_embed).float(), requires_grad=False)
            self.poi_view = nn.ModuleDict({
                'text_embed_mat': poi_embed_mat,
                'text_embed_layer': nn.Sequential(nn.LayerNorm(poi_embed.shape[1]),
                                                  nn.Linear(poi_embed.shape[1], d_model)),
                'poi_neighbors_att': Attention(d_model),
                'nearpoint_weight_layer': nn.Linear(d_model, d_model, bias=False),
                'OD_weight_layer': nn.Linear(d_model, d_model, bias=False),
                'merge_norm': nn.Sequential(nn.BatchNorm1d(d_model),
                                            nn.ReLU(inplace=True)), 
                'index_embed_layer': nn.Sequential(nn.Embedding(poi_embed.shape[0], embed_size),
                                                   nn.LayerNorm(embed_size),
                                                   nn.Linear(embed_size, d_model)),
                'seq_encoder': TrajMixerModel2(d_model=d_model, n_layer=2, d_intermediate=0, aux_feature_size=0, 
                                               d_inner=d_inner if d_model<=d_inner else d_model, # TODO
                                               block_type="Mamba2", device=device, dtype=torch.float32) 
            })

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def cal_traj_h(self, norm_spatial, temporal, road, aux_features, valid_lens, 
                   valid_indices=None, seq_idx=None, cu_seqlens=None):
        """Calculate trajectories' embedding vectors given their spatio-temporal features.
        The detailed definition of trajectories' spatial and temporal features can be refered to `data.py`.

        Args:
            norm_patial (FloatTensor): trajectories' spatial features, with shape (B, L, F_s).
            temporal (FloatTensor): trajectories' temporal features, with shape (B, L, F_t).
            road (LongTensor): trajectories' road features, with shape (B, L).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.
            aux_features (FloatTensor): trajectories' high-order features, with shape (B, L, F_h) or None.

        Returns:
            FloatTensor: the embedding vectors of this batch of trajectories, with shape (B, E).
        """
        B, L = norm_spatial.size(0), norm_spatial.size(1)
        
        spatial_e = self.traj_view['spatial_embed_layer'](norm_spatial)  # (B, L, E)

        temporal_tokens = [self.traj_view['temporal_embed_modules'][i](temp_token)
                        for i, temp_token in enumerate(tokenize_timestamp(temporal))]
        fine_temporal_e = self.traj_view['fine_temporal_embed_layer'](torch.cat(temporal_tokens[-2:], -1))
        coarse_temporal_e = self.traj_view['coarse_temporal_embed_layer'](torch.cat(temporal_tokens[:-2], -1))
        # Temporal values lower than 0 stands for feature mask.
        fine_temporal_e = fine_temporal_e.masked_fill(temporal[..., :1] < 0, 0)
        coarse_temporal_e = coarse_temporal_e.masked_fill(temporal[..., :1] < 0, 0)

        if self.use_road_in_traj_view:
            road_e = self.traj_view['road_index_embed_layer'](road)
            coarse_part = road_e + coarse_temporal_e
        else:
            coarse_part = coarse_temporal_e

        traj_h = torch.cat([coarse_part, spatial_e + fine_temporal_e], dim=-1)#+ pos_encoding
        
        traj_h = rearrange(traj_h, "b s ... -> (b s) ...")[valid_indices]
        aux_features = rearrange(aux_features, "b s ... -> (b s) ...")[valid_indices]
        traj_h = self.traj_view['seq_encoder'](traj_h, aux_features.unsqueeze(0), cu_seqlens[-1], seq_idx, cu_seqlens)
        traj_h = pad_input(traj_h, valid_indices, B, L).sum(1) / valid_lens.unsqueeze(-1)#repeat(valid_lens, 'B -> B 1')

        return traj_h

    def forward(self, input_seq, valid_lens):
        """
        Args:
            input_seq (FloatTensor): batch of trajectory features, with shape (B, L, F).
            valid_lens (LongTensor): valid lengths of trajectories in this batch.

        Returns:
            Tensor: embedding vectors for this batch of trajectories, with shape (B, E).
        """
        B, L, _ = input_seq.shape
        batch_mask = get_batch_mask(B, L, valid_lens)
        seq_idx, valid_indices = pack_input(repeat(torch.arange(end=B,dtype=torch.int32, device=valid_lens.device), 
                                "B -> B L", L=L), ~batch_mask)
        cu_seqlens = F.pad(torch.cumsum(valid_lens, dim=0, dtype=torch.int32), (1, 0))

        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        high_order_features = input_seq[:, :, COL_I['high_order_features']]
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        
        traj_h = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, valid_indices, seq_idx.unsqueeze(0), cu_seqlens)

        return traj_h

    def loss(self, input_seq, valid_lens):
        """Calcualte the pre-training loss and get high-order features calcualtion time.

        Args:
            Same as the forward function.

        Returns:
            FloatTensor: the pre-training loss value of this batch.
        """
        B, L, _ = input_seq.shape
        batch_mask = get_batch_mask(B, L, valid_lens)
        batch_arr = torch.arange(end=B,dtype=torch.int32,device=valid_lens.device)
        seq_idx, valid_indices = pack_input(repeat(batch_arr, "B -> B L", L=L), ~batch_mask)
        cu_seqlens = F.pad(torch.cumsum(valid_lens, dim=0, dtype=torch.int32), (1, 0))

        s_time = time.time()
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        high_order_features = input_seq[:, :, COL_I['high_order_features']]
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        e_time = time.time()
        
        # Trajectory (spatio-temporal) view.
        traj_h = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, valid_indices, seq_idx.unsqueeze(0), cu_seqlens)

        time_denom = temporal[..., 1][batch_arr, valid_lens-1].unsqueeze(-1).clamp_min(1e-6)
        time_ratio = temporal[..., 1] / time_denom #(B,L)
        # Road view.
        road_h = self.cal_road_h(road, valid_lens, time_ratio, batch_arr, 
                                 valid_indices, seq_idx.unsqueeze(0), cu_seqlens, batch_mask=batch_mask)
        
        # CLIP loss.
        traj_h = traj_h / traj_h.norm(dim=1, keepdim=True)
        road_h = road_h / road_h.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logit_road = logit_scale * traj_h @ road_h.t()

        label = torch.arange(B).long().to(input_seq.device)
        loss_road = (self.cross_entropy(logit_road, label) + self.cross_entropy(logit_road.t(), label)) / 2
        loss = self.road_weight * loss_road

        if self.use_poi_view and self.poi_weight:
            # POI view.
            poi_h = self.cal_poi_h(spatial, valid_lens, time_ratio, batch_arr,
                                   valid_indices, seq_idx.unsqueeze(0), cu_seqlens, batch_mask=batch_mask)
            poi_h = poi_h / poi_h.norm(dim=1, keepdim=True)
            logit_poi = logit_scale * traj_h @ poi_h.t()
            loss_poi = (self.cross_entropy(logit_poi, label) + self.cross_entropy(logit_poi.t(), label)) / 2
            loss = loss + self.poi_weight * loss_poi
        
        return loss, (e_time-s_time)
    
    def embed_road_ids_for_road_view(self, road):
        if self.use_textual_road_embed:
            return self.road_view['text_embed_layer'](self.road_view['text_embed_mat'](road))
        return self.road_view['road_id_embed_layer'](road)

    def cal_road_h(self, road, valid_lens, time_ratio, batch_arr, 
                   valid_indices=None, seq_idx=None, cu_seqlens=None, batch_mask=None):
        
        B, L = road.shape
        road_text_e = self.embed_road_ids_for_road_view(road) # (B,L,D)
        order = torch.arange(L, device=road.device).float().unsqueeze(0).expand(B, L)
        order_denom = (valid_lens - 1).clamp_min(1).float().unsqueeze(-1)
        order_ratio = order / order_denom
        order_e = self.road_view['order_embed_layer'](self.road_view['order_embed_module'](order_ratio))
        time_e = self.road_view['time_embed_layer'](self.road_view['time_embed_module'](time_ratio))
        road_text_e = road_text_e + order_e + time_e
        
        valid_neighbors_num = self.road_neighbors_num[road] # (B,L)
        N = valid_neighbors_num.max() # batch内路段的最大邻居数
        neighbors_mask = torch.arange(end=N, device=valid_lens.device).unsqueeze(0).unsqueeze(0) >= valid_neighbors_num.unsqueeze(-1) # (B, L, N) 
        neighbors_text_e = self.embed_road_ids_for_road_view(self.road_neighbors[road][:,:,:N]) # (B,L,N,D) 
        neighbors_weight = self.road_view['road_neighbors_att'](road_text_e, neighbors_text_e, neighbors_mask, batch_mask) # (B,L,N)
        static_neighbors_weight = self.road_neighbors_weight[road][:,:,:N]
        neighbors_weight = neighbors_weight + self.road_transW_lambda * static_neighbors_weight[...,1]
        neighbors_text_e = torch.matmul(neighbors_weight.unsqueeze(2), neighbors_text_e).squeeze(2) # (B,L,1,N) (B,L,N,E) -> (B,L,E)

        OD_text_e = (1-time_ratio).unsqueeze(-1) * road_text_e[:,0,:].unsqueeze(1) + time_ratio.unsqueeze(-1) * road_text_e[batch_arr, valid_lens-1].unsqueeze(1) #(B,L,E)

        merged_road_text_e = self.road_view['neighbors_weight_layer'](neighbors_text_e) + self.road_view['OD_weight_layer'](OD_text_e)

        road_h = road_text_e + self.road_view['merge_norm'](merged_road_text_e.permute(0,2,1)).permute(0,2,1) # (B,L,D)
        
        road_h = rearrange(road_h, "b s ... -> (b s) ...")[valid_indices]
        road_h = self.road_view['seq_encoder'](road_h, None, cu_seqlens[-1], seq_idx, cu_seqlens)
        road_h = pad_input(road_h, valid_indices, B, L).sum(1) / valid_lens.unsqueeze(-1)
        
        return road_h

    def cal_poi_h(self, spatial, valid_lens, time_ratio, batch_arr,
                  valid_indices=None, seq_idx=None, cu_seqlens=None, batch_mask=None):
        B, L, _ = spatial.shape

        nearest_poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) - spatial.unsqueeze(2)) ** 2).sum(-1).argmin(dim=-1) # (B,L)
        
        poi_index_e = self.poi_view['index_embed_layer'](nearest_poi)
        poi_text_e = self.poi_view['text_embed_layer'](self.poi_view['text_embed_mat'](nearest_poi))

        # 融合邻居POI的文本信息
        valid_neighbors_num = self.poi_neighbors_num[nearest_poi] # (B,L)
        N = valid_neighbors_num.max() # batch内POI的最大邻居数
        neighbors_mask = torch.arange(end=N, device=valid_lens.device).unsqueeze(0).unsqueeze(0) >= valid_neighbors_num.unsqueeze(-1) # (B, L, N)  
        neighbors_text_e = self.poi_view['text_embed_layer'](self.poi_view['text_embed_mat'](self.poi_neighbors[nearest_poi][:,:,:N])) # (B,L,N,D) 
        neighbors_weight = self.poi_view['poi_neighbors_att'](poi_text_e, neighbors_text_e, neighbors_mask, batch_mask) # (B,L,N)
        neighbors_weight = neighbors_weight + self.poi_distW_lambda * self.poi_neighbors_distW[nearest_poi][:,:,:N]
        nearpoint_merge = torch.matmul(neighbors_weight.unsqueeze(2), neighbors_text_e).squeeze(2) # (B,L,1,N) (B,L,N,E) -> (B,L,E)
        
        OD_merge = (1-time_ratio).unsqueeze(-1) * poi_text_e[:,0,:].unsqueeze(1) + time_ratio.unsqueeze(-1) * poi_text_e[batch_arr, valid_lens-1].unsqueeze(1) #(B,L,E)
        
        merged_poi_text_e = self.poi_view['nearpoint_weight_layer'](nearpoint_merge) + self.poi_view['OD_weight_layer'](OD_merge)
        
        poi_h = poi_index_e + poi_text_e + self.poi_view['merge_norm'](merged_poi_text_e.permute(0,2,1)).permute(0,2,1) 
        poi_h = rearrange(poi_h, "b s ... -> (b s) ...")[valid_indices]
        poi_h = self.poi_view['seq_encoder'](poi_h, None, cu_seqlens[-1], seq_idx, cu_seqlens)
        poi_h = pad_input(poi_h, valid_indices, B, L).sum(1) / valid_lens.unsqueeze(-1)

        return poi_h

    def construct_poi_neighbors(self): 
        '''
        input poi_coors pois_dist_thres poi_max_neighbors
        return poi_neighbors poi_neighbors_num poi_neighbors_distW
        '''
        dist_m = cal_tensor_geo_distance(self.poi_coors.unsqueeze(0), self.poi_coors.unsqueeze(1))
        dist_m = dist_m.fill_diagonal_(np.inf) # mask距离矩阵对角线（自身）
        neighbors_id = []
        neighbors_dist = []
        neighbors_num = []
        for i in range(len(self.poi_coors)):
            neighbors_indices = torch.where(dist_m[i] <= self.pois_dist_thres)[0]
            if len(neighbors_indices) > self.poi_max_neighbors:
                value, indices = torch.topk(dist_m[i][neighbors_indices], self.poi_max_neighbors, largest=False)
                neighbors_id.append(neighbors_indices[indices])
                neighbors_dist.append(value)
                neighbors_num.append(self.poi_max_neighbors)
            else:    
                neighbors_id.append(neighbors_indices)
                neighbors_dist.append(dist_m[i][neighbors_indices])
                neighbors_num.append(len(neighbors_indices))
        
        self.poi_neighbors = nn.Parameter(pad_sequence(neighbors_id, batch_first=True), requires_grad=False) 
        self.poi_neighbors_num = nn.Parameter(torch.tensor(neighbors_num), requires_grad=False)
        assert self.poi_neighbors_num.min() > 0, "pois_dist_thres set Too small!"
        
        poi_neighbors_distW = pad_sequence(neighbors_dist, batch_first=True)
        poi_neighbors_distW = torch.exp(-poi_neighbors_distW / poi_neighbors_distW.max(dim=-1, keepdim=True).values).masked_fill(get_batch_mask(self.poi_neighbors.shape[0], self.poi_max_neighbors, self.poi_neighbors_num), 0)
        
        self.poi_neighbors_distW = nn.Parameter((poi_neighbors_distW / poi_neighbors_distW.sum(dim=-1,keepdim=True)).float(), requires_grad=False)
    
    def forward_w_processtime_cal(self, input_seq, valid_lens):
        '''Calculate trajectories' embedding vectors given their spatio-temporal features and get process time of batch.
        '''
        start_time = time.time()
        B, L, _ = input_seq.shape
        batch_mask = get_batch_mask(B, L, valid_lens)
        seq_idx, valid_indices = pack_input(repeat(torch.arange(end=B,dtype=torch.int32, device=valid_lens.device), 
                                "B -> B L", L=L), ~batch_mask)
        cu_seqlens = F.pad(torch.cumsum(valid_lens, dim=0, dtype=torch.int32), (1, 0))

        s_time = time.time()
        spatial = input_seq[:, :, COL_I['spatial']]  # (B, L, 2)
        temporal = input_seq[:, :, COL_I['temporal']]  # (B, L, 2)
        road = input_seq[:, :, COL_I['road']].long() # (B,L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        high_order_features = input_seq[:, :, COL_I['high_order_features']]
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        e_time = time.time()
        
        traj_h = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, valid_indices, seq_idx.unsqueeze(0), cu_seqlens) # pos_encoding, batch_mask
        end_time = time.time()

        return traj_h, (end_time - start_time - (e_time-s_time)), (e_time-s_time)
    
    def forward_on_search_mode(self, input_seq, valid_lens):
        return self.forward_w_processtime_cal(input_seq, valid_lens)