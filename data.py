import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
from collections import Counter
from einops import repeat, rearrange
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from rdp import rdp
import utils


TRAJ_ID_COL = 'trip'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'
DT_COL = 'delta_t'
ROAD_COL = 'road'
V_COL = "speed"
ACC_COL = "acc"
ANGLE_COL = "courseAngle"
COL_I = {
    "spatial": [0, 1],
    "temporal": [2, 3],
    "road": 4,
    "high_order_features": [5,6,7]
}
FEATURE_PAD = 0
MIN_TRIP_LEN = 5
MAX_TRIP_LEN = 120
LONG_TRIP_WINDOW_STRIDE = MAX_TRIP_LEN // 2
BASE_TIME = pd.to_datetime('2018-09-29 00:00:00') # 时间戳起点重设，避免时间戳与其他特征一起转为torch.float32时出现精度损失

RAW_DATA_DIR = "./raw_data"
PROCESSED_DATA_DIR = "./processed_data"
TRAJ_META_DIR = os.path.join(PROCESSED_DATA_DIR, "traj_meta")
ROAD_META_DIR = os.path.join(PROCESSED_DATA_DIR, "road_meta")
SEARCH_META_DIR = os.path.join(PROCESSED_DATA_DIR, "search_meta")


class DataPreprocessor:
    def __init__(self, city):
        """
        Args:
            city (str): the city name to which trajectories belong.
            
        """
        self.city = city
        self.raw_data_dir = RAW_DATA_DIR if 'small' not in city else "./samples"
        self.dataset_category = ['train', 'valid', 'test']
        self.base_timestamp = get_last_Monday(BASE_TIME)
        utils.create_if_noexists(PROCESSED_DATA_DIR)
    
    def preprocess_traj(self, output_suffix=''):
        
        utils.create_if_noexists(os.path.join(TRAJ_META_DIR))
        
        for category in self.dataset_category:
            data_filename = self.city + '_' + category + '.h5'
            output_filename = self.city + '_' + category + output_suffix + '.h5'
            trajs_df = pd.read_hdf(os.path.join(self.raw_data_dir, data_filename), key="trips") if 'small' not in self.city \
                                            else pd.read_hdf(os.path.join(self.raw_data_dir, self.city + '.h5'), key="trips")
            trajs_df['timestamp'] = trajs_df['time'].apply(lambda x: x.timestamp())

            # Keep short trips directly and split long trips into windows instead
            # of dropping them. Each window is still bounded by MAX_TRIP_LEN,
            # which keeps the model input contract unchanged.
            preprocessed_trajs =[]
            num_kept, num_split, num_dropped, num_windows = 0, 0, 0, 0
            for trip_id, group in tqdm(trajs_df.groupby(TRAJ_ID_COL), desc='Preprocessing trips', total=len(trajs_df[TRAJ_ID_COL].unique()), ncols=70):
                if group.isna().any().any() or group.shape[0] < MIN_TRIP_LEN:
                    num_dropped += 1
                    continue

                windows = self.split_trip_into_windows(group, trip_id)
                if len(windows) > 1:
                    num_split += 1
                else:
                    num_kept += 1

                for window in windows:
                    window = self.cal_high_order_features(window)
                    preprocessed_trajs.append(window)
                    num_windows += 1

            preprocessed_trajs = pd.concat(preprocessed_trajs)
            preprocessed_trajs.to_hdf(os.path.join(TRAJ_META_DIR, output_filename), key="trips")
            print(
                f"Preprocessed {output_filename}: kept={num_kept}, split={num_split}, "
                f"dropped={num_dropped}, windows={num_windows}"
            )

    @staticmethod
    def split_trip_into_windows(group, trip_id):
        trip_len = group.shape[0]
        if trip_len <= MAX_TRIP_LEN:
            return [group.copy().reset_index(drop=True)]

        starts = list(range(0, trip_len - MAX_TRIP_LEN + 1, LONG_TRIP_WINDOW_STRIDE))
        tail_start = trip_len - MAX_TRIP_LEN
        if starts[-1] != tail_start:
            starts.append(tail_start)

        windows = []
        for window_i, start in enumerate(starts):
            window = group.iloc[start:start + MAX_TRIP_LEN].copy().reset_index(drop=True)
            window[TRAJ_ID_COL] = f"{trip_id}__win{window_i:03d}"
            windows.append(window)

        return windows
    
    @staticmethod
    def cal_high_order_features(group):
        group[DT_COL] = group[T_COL] - group[T_COL].iloc[0]
                    
        ST_features = group[[X_COL,Y_COL,T_COL]].to_numpy()

        time_diff = ST_features[1:, -1] - ST_features[:-1, -1]
        time_diff = np.where(time_diff != 0, time_diff, 1) # 除数不能为0
        
        dist = utils.cal_geo_distance(ST_features[:-1, :-1], ST_features[1:, :-1])
        group['distance'] = np.insert(dist, 0, values=-1) # 与前一个点的距离

        speed = dist / time_diff
        group['speed'] = speed = np.append(speed, values=0) # 终点速度置零 -> 停止

        acc = (speed[1:] - speed[:-1]) / time_diff
        group['acc'] = np.append(acc, values=0)

        courseAngle = utils.cal_courseAngle(ST_features[:-1, :-1], ST_features[1:, :-1])
        group['courseAngle'] = np.append(courseAngle, values=0)

        return group


    def compress_traj(self, dataset_mode='train', pred_len=0, input_suffix='', output_suffix=''):
        data_filename = self.city + '_' + dataset_mode + input_suffix
        output_filename = self.city + '_' + dataset_mode + output_suffix
        trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR,data_filename+'.h5'),key="trips")

        compressed_trajs =[]
        for _, group in tqdm(trajs_df.groupby(TRAJ_ID_COL), desc='Compressing trips', total=len(trajs_df[TRAJ_ID_COL].unique()), ncols=70):
            group = self.gen_compressed_traj(group, pred_len)

            compressed_trajs.append(group)
            
        compressed_trajs = pd.concat(compressed_trajs)
        compressed_trajs.to_hdf(os.path.join(TRAJ_META_DIR,output_filename+'_compressed_keep-{0}.h5'.format(pred_len)),key="trips")
    
    def gen_compressed_traj(self, group, pred_len):
        '''
            1. 删除多余的停留点
            2. 同一路段上的轨迹点，删除与前一点相比速度和加速度无明显变化的点
            3. 最后的'pred_len'个点不参与判断和压缩！——涉及的某些下游任务会删除原始轨迹中的最后X个点
            4. 更新轨迹的高阶特征！
        '''
            
        drop_index = self.find_intermediate_stay_point(group, pred_len)
        
        # 检查剩余点个数——存在一些位置全程不变的轨迹！
        if len(group) - len(drop_index) == pred_len+1: # 除保留点外只剩下轨迹起始点
            if len(drop_index):
                drop_index.pop()
            group.drop(drop_index, inplace = True)
            group = group.reset_index(drop=True) 
            return group
        
        group.drop(drop_index, inplace = True)
        group = group.reset_index(drop=True) 
        
        drop_index = self.find_steady_pace_point(group, pred_len)
        group.drop(drop_index, inplace = True)
        group = group.reset_index(drop=True) 
        
        group = self.cal_high_order_features(group)

        return group
    
    @staticmethod
    def find_intermediate_stay_point(group, pred_len):
        last_index_for_compress = group.index[-pred_len-1] if pred_len else group.index[-1]
            
        drop_index = []
        new_stay = True
        for index in group[group.distance == 0].index: # 删除多余的中间停留点，然后更新起始停止点的加速度
            if new_stay: # 进入新的停滞区间
                start = index - 1
                if start >= last_index_for_compress: break
                new_stay = False
            
            if group.loc[index,"speed"]: # 有速度了，停滞区间结束
                if last_index_for_compress < index-1: break
                new_stay = True
                if start+1 != index: # 起始停止点的加速度需要更新
                    group.loc[start,"acc"] = (group.loc[index,"speed"]-group.loc[start,"speed"]) / (group.loc[index,DT_COL]-group.loc[start,DT_COL])
            elif index <= last_index_for_compress:
                drop_index.append(index)

        return drop_index
    
    @staticmethod
    def find_steady_pace_point(group, pred_len):
        # 同一路段上的轨迹点，删除与前一点相比速度和加速度无明显变化的点（变化小于1.5/小于0.3）
        group["road_diff"] = group[ROAD_COL].diff(1)
        
        # 确保此时的轨迹终点不被drop！
        if not pred_len: 
            last_index_for_compress = group.index[-2]
        else: last_index_for_compress = group.index[-pred_len-1]
        
        drop_index = []
        index_for_update = []
        new_road = True
        for index in group[group.road_diff == 0].index:
            if index > last_index_for_compress: # 已超过轨迹可压缩部分的末尾
                break
            
            if new_road: # 进入新的路段
                pre_index = index-1 # 保存起始index
                index_for_update.append(pre_index)
                new_road = False
            
            if abs(group.loc[index,"speed"]-group.loc[pre_index,"speed"])<1.5 and abs(group.loc[index,"acc"]-group.loc[pre_index,"acc"])<0.3:
                drop_index.append(index) # 当前点需删除
            else: # 当前点需保留，并作为后续轨迹点的比较目标
                index_for_update.append(index)
                if index != last_index_for_compress: # index后还有可能需删除的点
                    pre_index = index
            
            if group.loc[index+1,"road_diff"]: # index是当前路段上最后一个轨迹点
                new_road = True
            
        group.drop(columns=['road_diff'], inplace=True)

        return drop_index


    def construct_road_neighbor_sets(self,):
        
        utils.create_if_noexists(os.path.join(ROAD_META_DIR))
        
        road_df = pd.read_hdf(os.path.join(self.raw_data_dir, self.city+'.h5'), key='road_info')
        num_road = int(road_df['road'].max() + 1)
        road_neighbors_id = []
        road_neighbors_lenW = []
        road_neighbors_num = []
        for road in tqdm(range(num_road), desc='Constructing road neighbor sets', ncols=70):
            group = road_df[road_df['road']==road]
            o, d = group['o'].iloc[0], group['d'].iloc[0]
            neighbor_roads = road_df[(road_df['o']==o) | (road_df['o']==d) | \
                                            (road_df['d']==o) | (road_df['d']==d)]
            neighbor_roads = neighbor_roads.drop(neighbor_roads[neighbor_roads['road']==road].index) # 不考虑自连通
            neighbor_roads_lenW = neighbor_roads['length'].values / neighbor_roads['length'].sum()
            road_neighbors_id.append(neighbor_roads['road'].to_numpy())
            road_neighbors_lenW.append(neighbor_roads_lenW)
            road_neighbors_num.append(len(road_neighbors_id[-1]))
        road_neighbors_id = pad_batch(road_neighbors_id).astype(np.int64) 
        road_neighbors_lenW = pad_batch(road_neighbors_lenW) 
        road_neighbors_num = np.array(road_neighbors_num)
        
        # 根据轨迹数据集获取相邻路段间转移概率
        trans_p = np.zeros((road_neighbors_id.shape[0], road_neighbors_id.shape[1]+1)) # 最后一列存储自身到自身的转移概率
        trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, self.city + '_train.h5'),key="trips")
        for _, group in tqdm(trajs_df.groupby(TRAJ_ID_COL), desc='Calculating frequency of roads transfer', total=len(trajs_df[TRAJ_ID_COL].unique()), ncols=70):
            road = group["road"].values
            start_points, end_points = road[:-1], road[1:] 
            whole_trans_times = len(start_points)

            indices = np.where(start_points!=end_points)[0].tolist() # 相邻轨迹点所在路段不同的下标
            remain_indices = list(set(range(whole_trans_times))-set(indices)) # 相邻轨迹点所在路段相同的下标
            np.add.at(trans_p, (start_points[remain_indices], -1), 1) # 增加自身到自身的转移频率
            
            neighbors_sets = road_neighbors_id[start_points[indices]] # 各"起点"的邻居路段集合
            neighbors = end_points[indices] # 各"终点"做要找的邻居
            neighbors_indices = np.argmax(neighbors_sets==neighbors[:, np.newaxis], axis=1) # 使用 np.argmax 找到第一个匹配的下标
            np.add.at(trans_p, (start_points[indices], neighbors_indices), 1) # 增加自身到邻居路段的转移频率
        road_trans_sum = trans_p.sum(axis=-1, keepdims=True)
        road_trans_sum[road_trans_sum == 0] = 1 # 对于和为0的行，填充1
        trans_p = trans_p / road_trans_sum # 频率转概率
        road_neighbors_weight = np.stack([road_neighbors_lenW, trans_p[:, :-1]], axis=-1)
        
        np.savez(os.path.join(ROAD_META_DIR,self.city+'_road_neighbors_info'),road_neighbors_sets=road_neighbors_id, road_neighbors_num=road_neighbors_num, road_neighbors_weight=road_neighbors_weight)


    def construct_STS_meta(self, test_traj_df_name, num_target=1000, num_negative=5000, same_OD_thres=50, 
                           gen_indices=True, only_indices=False, **kwargs):
        
        similar_trips_df = pd.read_hdf(os.path.join(SEARCH_META_DIR, self.city+"_test_SimTrips_2025May-"+ str(same_OD_thres) +".h5"), key='trips')
        
        # load原始的数据
        test_trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, self.city+'_test.h5'), key='trips')
        # 测试集和相似轨迹id匹配文件中的trip_id顺序一致
        assert list(test_trajs_df.groupby(TRAJ_ID_COL).groups.keys()) == similar_trips_df["query_trip_id"].tolist()
        test_trajs_df[T_COL] = test_trajs_df[T_COL] - self.base_timestamp

        if "compressed" in test_traj_df_name:
            compressed_test_trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, test_traj_df_name+'.h5'), key='trips')
            assert list(compressed_test_trajs_df.groupby(TRAJ_ID_COL).groups.keys()) == similar_trips_df["query_trip_id"].tolist()
            compressed_test_trajs_df[T_COL] = compressed_test_trajs_df[T_COL] - self.base_timestamp

        meta_dir = os.path.join(SEARCH_META_DIR, test_traj_df_name)
        utils.create_if_noexists(meta_dir)

        try:
            all_meta = np.load(os.path.join(meta_dir, f"all_meta_2025May-{same_OD_thres}.npz"), allow_pickle=True)
            all_targets = all_meta["targets"]
            all_queries = all_meta["queries"]
            similar_trip_not_self_indices = all_meta["similar_trip_not_self_indices"].tolist()
        except:
            all_queries = []
            all_targets = []
            similar_trip_not_self_indices = []
            for index, (trip_id, group) in enumerate(tqdm(test_trajs_df.groupby(TRAJ_ID_COL), desc='Gathering similar trips', total=len(test_trajs_df[TRAJ_ID_COL].unique()), ncols=70)):
                similar_trip_id = similar_trips_df.loc[index, 'target_trip_id'] 
                if similar_trip_id == trip_id:
                    group = group.reset_index(drop=True) # 重置索引
                    query, target = group[::2].reset_index(drop=True), group[1::2].reset_index(drop=True)
                    query, target = self.cal_high_order_features(query), self.cal_high_order_features(target)
                    if "compressed" in test_traj_df_name: # 压缩轨迹版
                        query = self.gen_compressed_traj(query,0) # 重新压缩
                        target = self.gen_compressed_traj(target,0)
                    query = query[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy()
                    target = target[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy()
                    all_queries.append(query)
                    all_targets.append(target)
                else:
                    similar_trip_not_self_indices.append(index)
                    if "compressed" in test_traj_df_name: # 压缩轨迹版
                        traj = compressed_test_trajs_df[compressed_test_trajs_df[TRAJ_ID_COL]==trip_id]
                        tgt_traj = compressed_test_trajs_df[compressed_test_trajs_df[TRAJ_ID_COL] == similar_trip_id]
                    else:
                        traj = group   
                        tgt_traj = test_trajs_df[test_trajs_df[TRAJ_ID_COL] == similar_trip_id]
                    all_queries.append(traj[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy())
                    all_targets.append(tgt_traj[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy())

            all_targets = np.array(all_targets, dtype=object)
            all_queries = np.array(all_queries, dtype=object)

            np.savez(os.path.join(meta_dir, f"all_meta_2025May-{same_OD_thres}.npz"), 
                     targets = all_targets,
                     queries = all_queries,
                     similar_trip_not_self_indices = np.array(similar_trip_not_self_indices, dtype=np.int64),
                    )
        print("Total number of similar trip not self:", len(similar_trip_not_self_indices))
        
        num_target = min(len(similar_trips_df) - 1, num_target)
        num_negative = min(len(similar_trips_df) - num_target, num_negative)
        # 按3:1的比例选择相似轨迹为自身/不为自身的轨迹
        sampled_trip_indices = []
        random.seed(10)
        sampled_trip_indices += random.sample(similar_trip_not_self_indices, num_target//4) if len(similar_trip_not_self_indices) >= num_target//4 else similar_trip_not_self_indices
        remain_trip_indices = list(np.delete(np.arange(len(similar_trips_df)), similar_trip_not_self_indices))
        random.seed(10)
        sampled_trip_indices += random.sample(remain_trip_indices, num_target - len(sampled_trip_indices))

        qry_trips = all_queries[sampled_trip_indices]
        tgt_trips = all_targets[sampled_trip_indices]
        qrytgt = np.concatenate((qry_trips,tgt_trips), axis=0, dtype=object)

        all_tgt_idx = np.array(similar_trips_df['target_trip_index']) # 可能存在重复值！一些轨迹既是自身的target也是另一条轨迹的target！
        neg_indices_list = []
        for sampled_trip_index in sampled_trip_indices:
            non_neg_indices = similar_trips_df.loc[sampled_trip_index, 'non_neg_indices']
            neg_indices = np.delete(np.arange(len(similar_trips_df)), non_neg_indices)
            neg_indices = np.random.choice(neg_indices, num_negative, replace=False)
            neg_indices_list.append(neg_indices)
        neg_indices_list = np.array(neg_indices_list)
        print("neg_indices shape: ", neg_indices_list.shape)

        if gen_indices:
            np.savez(os.path.join(SEARCH_META_DIR, f"{self.city}_test_sim_indices_2025May-{num_target}-{num_negative}-{same_OD_thres}"), 
                     qry_idx = np.array(sampled_trip_indices),
                     tgt_idx = np.array(similar_trips_df.loc[sampled_trip_indices, 'target_trip_index']),
                     neg_idx = neg_indices_list,
                    )
            if only_indices:
                return
            
        np.savez(os.path.join(meta_dir, f"qrytgt_negidx_2025May-{num_target}-{num_negative}-{same_OD_thres}"), 
                     qrytgt = qrytgt,
                     neg_indices = neg_indices_list
                    )
        print("Saved meta to", meta_dir)

    def Testset_SimTraj_Label(self, same_OD_thres=50, neighbor_area_radius=500, **kwargs):
        '''
        1. 基于完整轨迹构造相似轨迹label
        2. 根据起点pair和终点pair之间的距离（范围）确定候选轨迹集和负样本轨迹集
        3. 基于GPS和road seg确定最相似的轨迹
        '''
        filename = self.city+"_test_SimTrips_2025May-"+ str(same_OD_thres) +".h5"
        print(f"Need File: {filename}")
        if os.path.exists(os.path.join(SEARCH_META_DIR, filename)):
            print("File exists.")
            return
        
        utils.create_if_noexists(os.path.join(SEARCH_META_DIR))

        test_trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, self.city+'_test.h5'),key="trips")
        trip_ori = []
        trip_dest = []
        trajs = []
        for trip_id, group in tqdm(test_trajs_df.groupby(TRAJ_ID_COL), desc='Gathering OD of trips', total=len(test_trajs_df[TRAJ_ID_COL].unique()), ncols=70):
            traj = group[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy()
            trajs.append(traj)
            trip_ori.append(traj[0, :2])
            trip_dest.append(traj[-1, :2])   
        trajs = np.array(trajs, dtype=object)
        trip_ori = np.array(trip_ori)
        trip_dest = np.array(trip_dest)

        Ori_dist = utils.cal_geo_distance(trip_ori[np.newaxis, :], trip_ori[:, np.newaxis])
        Dest_dist = utils.cal_geo_distance(trip_dest[np.newaxis, :], trip_dest[:, np.newaxis])
        same_OD_trips = (Ori_dist <= same_OD_thres) & (Dest_dist <= same_OD_thres)
        neighbor_area_trips = (Ori_dist <= neighbor_area_radius) & (Dest_dist <= neighbor_area_radius)
        
        similar_trips = {'query_trip_id': np.array(list(test_trajs_df.groupby(TRAJ_ID_COL).groups.keys())),
                         'query_trip_index': np.arange(len(test_trajs_df[TRAJ_ID_COL].unique())),
                         'same_OD_trips_ids': [],
                         'same_OD_trips_indices': [],
                         'target_trip_id': [],
                         'target_trip_index': [],
                         'non_neg_ids':[],
                         'non_neg_indices':[]
                        }
        
        for index, trip_id in enumerate(tqdm(similar_trips['query_trip_id'], desc='Search similar trips', total=len(test_trajs_df[TRAJ_ID_COL].unique()), ncols=70)):
            same_OD_trips_indices = np.nonzero(same_OD_trips[index])[0]
            same_OD_trips_ids = similar_trips['query_trip_id'][same_OD_trips_indices]
            similar_trips['same_OD_trips_ids'].append(same_OD_trips_ids)
            similar_trips['same_OD_trips_indices'].append(same_OD_trips_indices)
            
            near_trips_label = neighbor_area_trips[index]
            similar_trips['non_neg_ids'].append(similar_trips['query_trip_id'][near_trips_label])
            similar_trips['non_neg_indices'].append(similar_trips['query_trip_index'][near_trips_label])
            
            if len(same_OD_trips_ids)==1:
                # print("No trips have close OD.")
                similar_trips['target_trip_id'].append(trip_id) # 选取自身做target
                similar_trips['target_trip_index'].append(index)
            else: # same_OD_thres取较小的值，因此当具有起点终点相似的轨迹时，选自身以外的与自身最相似的轨迹做target
                trip = trajs[index]
                dist_list = []
                road_diff_list = []
                for same_OD_trip_index in same_OD_trips_indices:
                    if same_OD_trip_index == index:
                        distance, path = fastdtw(trip[::2,:2], trip[1::2, :2], dist=utils.cal_geo_distance)
                        road_diff = len(list(set(trip[::2,4]) ^ set(trip[1::2, 4])))
                    else:
                        candidate_trip = trajs[same_OD_trip_index]
                        distance, path = fastdtw(trip[:,:2], candidate_trip[:, :2], dist=utils.cal_geo_distance)
                        road_diff = len(list(set(trip[:,4]) ^ set(candidate_trip[:, 4])))
                    dist_list.append(distance)
                    road_diff_list.append(road_diff)
                assert len(same_OD_trips_ids) == len(dist_list)
                dist_list = np.array(dist_list)
                road_diff_list = np.array(road_diff_list,dtype=np.int64)
                norm_dist = (dist_list - dist_list.min()) / (dist_list.max() - dist_list.min())
                norm_road_diff = (road_diff_list - road_diff_list.min()) / (road_diff_list.max()-road_diff_list.min() if road_diff_list.max()>road_diff_list.min() else 1)
                eval = np.sum([norm_dist, norm_road_diff], axis=0).tolist()
                min_dist_index = eval.index(min(eval))
                similar_trips['target_trip_id'].append(same_OD_trips_ids[min_dist_index])
                similar_trips['target_trip_index'].append(same_OD_trips_indices[min_dist_index])
            
        df = pd.DataFrame(similar_trips)
        df.to_hdf(os.path.join(SEARCH_META_DIR, filename), key="trips")


    def compress_traj_with_rdp(self, dataset_mode='train', pred_len=0):
        data_filename = self.city + '_' + dataset_mode
        trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR,data_filename+'.h5'),key="trips")

        compressed_trajs =[]
        for _, group in tqdm(trajs_df.groupby(TRAJ_ID_COL), desc='Compressing trips by RDP...', total=len(trajs_df[TRAJ_ID_COL].unique()), ncols=70):
            gps = group[[X_COL, Y_COL]]
            traj_mask = rdp(gps.iloc[:-pred_len] if pred_len else gps, 
                            epsilon=3e-6, algo="iter", return_mask=True)
            traj_mask[0], traj_mask[-1] = True, True # 确保起点、终点不被mask
            traj_mask += [True] * pred_len # 最后pred_len一定保留
            group = group[traj_mask]

            compressed_trajs.append(group)
            
        compressed_trajs = pd.concat(compressed_trajs)
        compressed_trajs.to_hdf(os.path.join(TRAJ_META_DIR,data_filename+'_RDPcompressed_keep-{0}.h5'.format(pred_len)),key="trips")


class TrajClipDataset(Dataset):
    """Dataset support class for TrajCLIP.

    Args:
        traj_df (pd.DataFrame): contains points of all trajectories.
        traj_ids (pd.Series): records the unique IDs of all trajectory sequences.
        spatial_border (list): coordinates indicating the spatial border: [[x_min, y_min], [x_max, y_max]].
    """

    def __init__(self, traj_df): 
        """
        Args:
            traj_df (pd.DataFrame): contains points of all trajectories.
        """
        super().__init__()
        self.traj_df = traj_df

        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique() 

        self.base_timestamp = get_last_Monday(BASE_TIME)
        self.traj_df[T_COL] = self.traj_df[T_COL] - self.base_timestamp # 重置时间戳起点

        self.valid_trajs = self.traj_df.groupby(TRAJ_ID_COL)

        '''用于在main.py初始化各模型，实际只会用train_dataset的统计数据'''
        spatial_border = self.traj_df[[X_COL, Y_COL]]
        self.spatial_border = [spatial_border.min().tolist(), spatial_border.max().tolist()]

        temporal_border = self.traj_df[[T_COL, DT_COL]]
        self.temporal_border = [temporal_border.min().tolist(), temporal_border.max().tolist()]

        high_order_feature_border = self.traj_df[[V_COL, ACC_COL, ANGLE_COL]]
        self.high_order_feature_border = [high_order_feature_border.min().tolist(), high_order_feature_border.max().tolist()]


    def use_partial_data(self, prop=0.2):
        assert 0.0 < prop < 1.0
        sampled_indices = np.sort(np.random.choice(len(self.traj_ids), size=int(prop*len(self.traj_ids)), replace=False))
        self.traj_ids = self.traj_ids[sampled_indices]
    
    def reset_data(self):
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

    def __len__(self):
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        one_traj = self.valid_trajs.get_group(self.traj_ids[index])
        return one_traj


class PretrainPadder:
    """Collate function for padding pre-training data.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        self.device = device

    def __call__(self, raw_batch):
        """Collate function for padding the raw batch of trajectory DataFrames into Tensors.

        Args:
            raw_batch (list): each item is a `pd.DataFrame` representing one trajectory.

        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
        """
        traj_batch, valid_lens = [], []
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy()
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device) 
        valid_lens = torch.tensor(valid_lens).long().to(self.device)

        return traj_batch, valid_lens


class DpPadder:
    """Collate function for padding destination prediction (DP) task data.
    """

    def __init__(self, device, pred_len, pred_cols):
        """
        Args:
            device (str): name of the device to put tensors on.
            pred_len (int): the length of the tail sub-trajectory to remove from the input trajectory.
            pred_cols (list): the columns to predict.
        """
        self.device = device
        self.pred_len = pred_len
        self.pred_cols = pred_cols

    def __call__(self, raw_batch):
        """
        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
            torch.FloatTensor: the ground truth of the DP task, i.e., features of the last trajectory point, 
            with shape (B, F).
        """
        traj_batch, valid_lens, label_batch = [], [], []
        for row in raw_batch:
            traj = row[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy()
            traj = traj[:-self.pred_len]
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)

            label = row.iloc[-1][self.pred_cols].to_numpy()
            label_batch.append(label)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        label_batch = torch.from_numpy(np.stack(label_batch, 0).astype(float)).float().to(self.device)

        return traj_batch, valid_lens, label_batch


class PretrainDatasetsUnion(Dataset):
    def __init__(self, teacher_embeds, compressed_dataset:TrajClipDataset):
        """
        Args:
            traj_df (pd.DataFrame): contains points of all trajectories.
        """
        super().__init__()
        assert len(compressed_dataset) == len(teacher_embeds)
        self.teacher_embeds = teacher_embeds
        self.compressed_dataset = compressed_dataset
    
    def __len__(self):
        return self.compressed_dataset.traj_ids.shape[0]

    def __getitem__(self, index):
        compressed_traj = self.compressed_dataset.__getitem__(index)
        teacher_traj_h = self.teacher_embeds[index]
        return compressed_traj, teacher_traj_h
    
class UnionPretrainPadder:
    """Collate function for padding pre-training data.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        self.device = device

    def __call__(self, raw_batch):
        """Collate function for padding the raw batch of trajectory DataFrames into Tensors.

        Args:
            raw_batch (list): each item is a `pd.DataFrame` representing one trajectory.

        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
        """
        compressed_traj_batch, compressed_valid_lens = [], []
        teacher_embeds_batch = []
        for row in raw_batch:
            compressed_traj, teacher_traj_h = row
            teacher_embeds_batch.append(teacher_traj_h)

            compressed_traj = compressed_traj[[X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]].to_numpy()
            compressed_valid_len = compressed_traj.shape[0]
            compressed_traj_batch.append(compressed_traj)
            compressed_valid_lens.append(compressed_valid_len)

        compressed_traj_batch = torch.from_numpy(pad_batch(compressed_traj_batch)).float().to(self.device)
        compressed_valid_lens = torch.tensor(compressed_valid_lens).long().to(self.device)
        teacher_embeds_batch = torch.from_numpy(np.array(teacher_embeds_batch)).float().to(self.device)
        
        return (compressed_traj_batch, compressed_valid_lens), teacher_embeds_batch


def fetch_task_padder(padder_name, device, padder_params):
    if padder_name == 'dp':
        task_padder = DpPadder(device, **padder_params)
    elif padder_name == "pretrain":
        task_padder = PretrainPadder(device)
    elif padder_name == "unionpretrain":
        task_padder = UnionPretrainPadder(device)
    elif padder_name == "search":
        task_padder = SearchPadder(device)
    else:
        raise NotImplementedError(f'No Padder named {padder_name}')

    return task_padder


def pad_batch(batch):
    """
    Pad the batch to the maximum length of the batch.

    Args:
        batch (list): the batch of arrays to pad, [(L1, F), (L2, F), ...] or [(L1), (L2), ...].

    Returns:
        np.array: the padded array.
    """
    max_len = max([arr.shape[0] for arr in batch])
    padded_batch = np.full((len(batch), max_len, batch[0].shape[-1]), FEATURE_PAD, dtype=float) if batch[0].ndim > 1 \
                    else np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    for i, arr in enumerate(batch):
        padded_batch[i, :arr.shape[0]] = arr

    return padded_batch

def get_last_Monday(start_time: pd.Timestamp):
    days_to_monday = (start_time.weekday() + 1) % 7  # 计算到最近的星期一的天数
    last_monday = start_time - pd.Timedelta(days=days_to_monday)  # 减去天数
    # 设置时间为 00:00
    last_monday_at_midnight = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    return last_monday_at_midnight.timestamp()


class TrajectorySearchTestdata:    
    @staticmethod
    def parse_label(length):
        qry_idx = list(range(int(length / 2)))
        tgt_idx = list(range(int(length / 2), length))
        return qry_idx, tgt_idx

    @staticmethod
    def cal_pres_and_labels(query, target, negs):
        """
        query: (N, d)
        target: (N, d)
        negs: (N, n, d)
        """
        num_queries = query.shape[0]
        num_targets = target.shape[0]
        num_negs = negs.shape[1]
        print("query: ", query.shape)
        print("target: ", target.shape)
        print("neg: ", negs.shape)
        assert num_queries == num_targets, "Number of queries and targets should be the same."

        query_t = repeat(query, 'nq d -> nq nt d', nt=num_targets)
        query_n = repeat(query, 'nq d -> nq nn d', nn=num_negs)
        target = repeat(target, 'nt d -> nq nt d', nq=num_queries)
        # negs = repeat(negs, 'nn d -> nq nn d', nq=num_queries)

        dist_mat_qt = np.linalg.norm(query_t - target, ord=2, axis=2)
        dist_mat_qn = np.linalg.norm(query_n - negs, ord=2, axis=2)
        dist_mat = np.concatenate([dist_mat_qt[np.eye(num_queries).astype(bool)][:, None], dist_mat_qn], axis=1)

        pres = -1 * dist_mat

        labels = np.zeros(num_queries)

        return pres, labels


def load_trajSearch_testdata(search_meta_dir, num_target=1000, num_negative=5000, same_OD_thres=50, **kwargs): 
    
    alltrajtgt = np.load(os.path.join(search_meta_dir, f"all_meta_2025May-{same_OD_thres}.npz"), allow_pickle=True)["targets"]
    qrytgt_negidx = np.load(os.path.join(search_meta_dir, f"qrytgt_negidx_2025May-{num_target}-{num_negative}-{same_OD_thres}.npz"), allow_pickle=True)
    hopqrytgt = qrytgt_negidx["qrytgt"]
    neg_indices = qrytgt_negidx["neg_indices"]
    return alltrajtgt, hopqrytgt, neg_indices


class TrajectorySearchDataset(Dataset):
    def __init__(self, trajs):
        super().__init__()
        self.trajs = trajs
        self.valid_trajs = self.trajs
    
    def compress_trajs_with_rdp(self):
        compressed_trajs = []
        for one_traj in tqdm(self.trajs, desc='Compressing trips by RDP...', total=len(self.trajs), ncols=70):
            traj_mask = rdp(one_traj[:, :2], epsilon=3e-6, algo="iter", return_mask=True)
            traj_mask[0], traj_mask[-1] = True, True
            one_traj = one_traj[traj_mask]
            compressed_trajs.append(one_traj)
        self.valid_trajs = np.array(compressed_trajs, dtype=object)

    def __len__(self):
        return self.valid_trajs.shape[0]

    def __getitem__(self, index):
        one_traj = self.valid_trajs[index]
        return one_traj

class SearchPadder:
    """Collate function for padding data of similar trajectory search.
    """

    def __init__(self, device):
        """
        Args:
            device (str): name of the device to put tensors on.
        """
        self.device = device

    def __call__(self, raw_batch):
        """Collate function for padding the raw batch of trajectory DataFrames into Tensors.

        Args:
            raw_batch (list): each item is a `pd.DataFrame` representing one trajectory.

        Returns:
            torch.FloatTensor: the padded batch of trajectory features, with shape (B, L, F).
            torch.LongTensor: the valid lengths of trajectories in the batch, with shape (B).
        """
        traj_batch, valid_lens = [], []
        feature_cols = [X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]
        for traj in raw_batch:
            if isinstance(traj, pd.DataFrame):
                traj = traj[feature_cols].to_numpy()
            valid_len = traj.shape[0]
            traj_batch.append(traj)
            valid_lens.append(valid_len)
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)

        return traj_batch, valid_lens


if __name__ == '__main__':
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test_search") 
    args = parser.parse_args()

    # Load the settings file, and save a backup in the cache directory.
    with open(os.path.join('settings', f'{args.settings}.json'), 'r') as fp:
        settings = json.load(fp)
    # Iterate through the multiple settings.
    for setting_i, setting in enumerate(settings):
        print(f'===SETTING {setting_i}/{len(settings)}===')
        SAVE_NAME = setting.get('save_name', None)

        preprocessor = DataPreprocessor(setting['dataset']['city'])
        traj_output_suffix = setting['dataset'].get('traj_output_suffix', '')
        preprocessor.preprocess_traj(output_suffix=traj_output_suffix)
        
        preprocessor.construct_road_neighbor_sets()
        
        # used in pretrain stage
        preprocessor.compress_traj(dataset_mode='train', pred_len=0, input_suffix=traj_output_suffix, output_suffix=traj_output_suffix)
        preprocessor.compress_traj(dataset_mode='test', pred_len=0, input_suffix=traj_output_suffix, output_suffix=traj_output_suffix)
        
        # used in downstream task stage(fine-tuning or without fine-tuning)
        preprocessor.compress_traj(dataset_mode='train', pred_len=5, input_suffix=traj_output_suffix, output_suffix=traj_output_suffix)
        preprocessor.compress_traj(dataset_mode='valid', pred_len=5, input_suffix=traj_output_suffix, output_suffix=traj_output_suffix)
        preprocessor.compress_traj(dataset_mode='test', pred_len=5, input_suffix=traj_output_suffix, output_suffix=traj_output_suffix)

        if 'test' in setting:
            preprocessor.Testset_SimTraj_Label(**setting['test']["search_data_params"])
            
            eval_dataset = os.path.basename(setting['dataset']['test_traj_df']).split(".")[0]
            # meta_dir = os.path.join(SEARCH_META_DIR, eval_dataset)
            # print('meta_dir:',meta_dir)

            preprocessor.construct_STS_meta(eval_dataset, **setting['test']["search_data_params"])