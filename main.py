import os
import sys
import json
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-s', '--settings', help='name of the settings file to use', type=str, default="local_test") 
parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default='6')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

import torch
from torch.utils.data import DataLoader

import utils
from data import TrajClipDataset, PretrainPadder, PretrainDatasetsUnion, UnionPretrainPadder, DpPadder, TrajectorySearchDataset, SearchPadder, \
    fetch_task_padder, load_trajSearch_testdata, X_COL, Y_COL, SEARCH_META_DIR, TRAJ_META_DIR, ROAD_META_DIR, DT_COL, ROAD_COL
from pipeline import pretrain_model, finetune_model, test_model, test_model_on_search
from models.trajectory_mamba import Trajectory_Mamba
from models.predictor import MlpPredictor
from models.TrajCLIP import TrajClip


SETTINGS_CACHE_DIR = os.environ.get('SETTINGS_CACHE_DIR', os.path.join('settings', 'cache'))
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', 'saved_model')
PRED_SAVE_DIR = os.environ.get('PRED_SAVE_DIR', 'predictions')
LOG_DIR = os.environ.get('LOG_DIR', '/root/autodl-tmp/log')


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_experiment_logging(log_name):
    utils.create_if_noexists(LOG_DIR)
    log_path = os.path.join(LOG_DIR, f'{log_name}.log')
    log_fp = open(log_path, 'a', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)
    print(f'==== Logging to {log_path} ====')
    return log_path


def main():
    setup_experiment_logging(args.settings)
    device = f'cuda:0' if torch.cuda.is_available() and args.cuda is not None else 'cpu'

    # This key is an indicator of multiple things.
    datetime_key = utils.get_datetime_key()
    print(f'====START EXPERIMENT, DATETIME KEY: {datetime_key} ====')

    # Load the settings file, and save a backup in the cache directory.
    with open(os.path.join('settings', f'{args.settings}.json'), 'r') as fp:
        settings = json.load(fp)
    utils.create_if_noexists(SETTINGS_CACHE_DIR)
    with open(os.path.join(SETTINGS_CACHE_DIR, f'{datetime_key}.json'), 'w') as fp:
        json.dump(settings, fp)

    # Iterate through the multiple settings.
    for setting_i, setting in enumerate(settings):
        print(f'===SETTING {setting_i}/{len(settings)}===')
        SAVE_NAME = setting.get('save_name', None)

        # Load and build training and testing datasets.
        train_traj_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, setting['dataset']['train_traj_df']), key='trips')
        train_dataset = TrajClipDataset(traj_df=train_traj_df)

        # Load road segments and, when enabled, POIs' coordinates and textual embeddings.
        road_embed_path = setting['dataset'].get('road_embed')
        road_embed = np.load(road_embed_path) if road_embed_path else None
        use_poi_view = setting.get('traj_clip', {}).get('use_poi_view', True)
        if use_poi_view:
            poi_df = pd.read_hdf(setting['dataset']['poi_df'], key='pois')
            poi_embed = np.load(setting['dataset']['poi_embed'])
            poi_coors = poi_df[[X_COL, Y_COL]].to_numpy()
        else:
            poi_embed, poi_coors = None, None

        # Load road segments' neighbor sets.
        road_neighbors = np.load(os.path.join(ROAD_META_DIR, setting['dataset']['road_neighbors']),allow_pickle=True)
        num_road = setting.get('num_road') or setting.get('pred_head', {}).get('output_size')
        if num_road is None and road_embed is not None:
            num_road = road_embed.shape[0]
        if num_road is None and "road_neighbors_sets" in road_neighbors:
            num_road = road_neighbors["road_neighbors_sets"].shape[0]
        if num_road is None:
            num_road = int(train_traj_df[ROAD_COL].max() + 1)
        num_road = int(num_road)

        # 设置随机种子
        if "seed" in setting:
            utils.setup_seed(setting["seed"])

        # Build the trajectory embedding model and the downstream prediction head.
        trajectory_mamba_model = Trajectory_Mamba(num_road=num_road, spatial_border=train_dataset.spatial_border, 
                                                  high_order_feature_border=train_dataset.high_order_feature_border, temporal_border=train_dataset.temporal_border,
                                                  device=device, **setting['trajectory_mamba']).to(device)
        traj_clip = TrajClip(road_embed=road_embed, num_road=num_road, road_neighbors=road_neighbors, 
                             poi_embed=poi_embed, poi_coors=poi_coors,
                             spatial_border=train_dataset.spatial_border, high_order_feature_border=train_dataset.high_order_feature_border, 
                             device=device, **setting['trajectory_mamba'], **setting['traj_clip']).to(device) 
        pred_head = MlpPredictor(spatial_border=train_dataset.spatial_border, **setting['pred_head']).to(device)
        size_all_mb = utils.cal_model_size(trajectory_mamba_model)
        print(f"Trajectory-Mamba Model size: {size_all_mb} MBytes.")
        size_all_mb = utils.cal_model_size(traj_clip)
        print(f"TrajClip Model size: {size_all_mb} MBytes.")


        if 'pretrain' in setting:
            # Pretrain the trajectory embedding model with self-supervised CLIP loss.
            teacher_prepared = False
            teacher_savename = setting['pretrain'].get("teacher_model_savename", None)
            if teacher_savename is not None:
                traj_clip.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{teacher_savename}_trajclip.pretrain'),
                                                    map_location=device))
                teacher_prepared = True

            if setting['pretrain'].get('load', False):
                # Load previously saved model parameters.
                PRETRAIN_SAVE_NAME = setting['pretrain'].get('pretrain_save_name', SAVE_NAME) # one pretrained model may correspond to multiple types of finetune. 
                if not teacher_prepared:
                    traj_clip.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{PRETRAIN_SAVE_NAME}_trajclip.pretrain'),
                                                        map_location=device))
                trajectory_mamba_model.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{PRETRAIN_SAVE_NAME}_trajectorymamba.pretrain'),
                                                     map_location=device))
            else: 
                # Pretrain trajectory embedding model (and teacher TrajClip)
                utils.create_if_noexists(MODEL_CACHE_DIR)
                compressed_pretrain_traj_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, setting['dataset']['compressed_pretrain_traj_df']), key='trips')
                compressed_pretrain_dataset = TrajClipDataset(traj_df=compressed_pretrain_traj_df)
                assert len(compressed_pretrain_dataset)==len(train_dataset)
                
                pretrain_dataset = partial(PretrainDatasetsUnion, compressed_dataset=compressed_pretrain_dataset)
                pretrain_dl_cfg = dict(setting['pretrain']['dataloader'])
                pretrain_shuffle = pretrain_dl_cfg.pop('shuffle', True)
                pretrain_dl_cfg.pop('drop_last', None)
                pretrain_padder = PretrainPadder(device=device, **setting['pretrain']['padder'])
                # Stage 2 MEC uses BatchNorm and fails on the final batch of size 1.
                pretrain_dataloader = partial(DataLoader,
                                                 collate_fn=UnionPretrainPadder(
                                                     device=device, **setting['pretrain']['padder']),
                                                 shuffle=pretrain_shuffle,
                                                 drop_last=True,
                                                 **pretrain_dl_cfg)
                
                teacher_dataloader = DataLoader(train_dataset,
                                                 collate_fn=pretrain_padder,
                                                 shuffle=pretrain_shuffle,
                                                 drop_last=True,
                                                 **pretrain_dl_cfg) if not teacher_prepared else None
                
                unshuffle_teacher_dataloader = DataLoader(train_dataset,
                                                          shuffle=False,
                                                          drop_last=False,
                                                          collate_fn=pretrain_padder,
                                                          **pretrain_dl_cfg)

                pretrain_model(model=trajectory_mamba_model, teacher_model=traj_clip, 
                                dataset=pretrain_dataset, dataloader=pretrain_dataloader, 
                                unshuffle_teacher_dataloader=unshuffle_teacher_dataloader, 
                                teacher_dataloader=teacher_dataloader, teacher_prepared=teacher_prepared, 
                                teacher_savename=os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_trajclip.pretrain'),
                                checkpoint_dir=setting['pretrain'].get('checkpoint', {}).get('dir'),
                                resume=setting['pretrain'].get('checkpoint', {}).get('resume', False),
                                save_name=SAVE_NAME,
                                **setting['pretrain']['config'])

                if setting['pretrain'].get('save', True):
                    # Save the pretrained model parameters.
                    utils.create_if_noexists(MODEL_CACHE_DIR)
                    if not teacher_prepared:
                        torch.save(traj_clip.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_trajclip.pretrain'))
                    torch.save(trajectory_mamba_model.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_trajectorymamba.pretrain'))

        if 'finetune' in setting:
            # Finetune the trajectory embedding model and the prediction head on downstream tasks.
            if setting['finetune'].get("check_teacher",False):
                print("\nFinetune Teacher model:")
                model = traj_clip
                model_name = 'trajclip'
                finetune_dataset = train_dataset
            else:
                print("\nFinetune Trajectory-Mamba:") 
                model = trajectory_mamba_model
                model_name = 'trajectorymamba'
                compressed_finetune_traj_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, setting['dataset']['compressed_finetune_traj_df']), key='trips')
                finetune_dataset = TrajClipDataset(traj_df=compressed_finetune_traj_df)

            if setting['finetune'].get('load', False):
                model.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_{model_name}.finetune')))
                pred_head.load_state_dict(torch.load(os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_predhead.finetune')))
            else:
                if setting['finetune'].get('prop', 1.0) < 1.0 :
                    print("Train set prop =", setting['finetune']['prop'])
                    finetune_dataset.use_partial_data(setting['finetune']['prop'])

                finetune_padder = fetch_task_padder(padder_name=setting['finetune']['padder']['name'],
                                                    device=device, padder_params=setting['finetune']['padder']['params'])
                finetune_dataloader = DataLoader(finetune_dataset, collate_fn=finetune_padder,
                                                 **setting['finetune']['dataloader'])
                
                if setting['dataset'].get('val_traj_df', False):
                    val_traj_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, setting['dataset']['val_traj_df']), key='trips')
                    val_dataset = TrajClipDataset(traj_df=val_traj_df)
                    val_dataloader = DataLoader(val_dataset, collate_fn=finetune_padder,
                                                    **setting['finetune']['val_dataloader'])
                else: val_dataloader = None
                valid_metrics_list = []
                
                if_denormalize = False
                pred_cols = None
                if isinstance(finetune_padder, DpPadder) and sorted(finetune_padder.pred_cols) == sorted([Y_COL, X_COL]): # 预测GPS
                    if_denormalize = True
                    pred_cols=finetune_padder.pred_cols

                finetune_model(model=model, pred_head=pred_head, dataloader=finetune_dataloader,
                               val_dataloader=val_dataloader, valid_metrics_list=valid_metrics_list, 
                               denormalize=if_denormalize, pred_cols=pred_cols, **setting['finetune']['config'])

                if setting['finetune'].get('save', True):
                    utils.create_if_noexists(MODEL_CACHE_DIR)
                    torch.save(model.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_{model_name}.finetune'))
                    torch.save(pred_head.state_dict(), os.path.join(MODEL_CACHE_DIR, f'{SAVE_NAME}_predhead.finetune'))
                if len(valid_metrics_list):
                    valid_metrics_df = pd.concat(valid_metrics_list, axis=1).T
                    utils.create_if_noexists(os.path.join(PRED_SAVE_DIR, SAVE_NAME))
                    valid_metrics_df.to_csv(os.path.join(PRED_SAVE_DIR, SAVE_NAME, 'valid_metrics.csv'), index=False)

        if 'test' in setting:
            # Test the model on downstream tasks.
            test_traj_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, setting['dataset']['test_traj_df']), key='trips')
            test_dataset = TrajClipDataset(traj_df=test_traj_df)
            test_padder = fetch_task_padder(padder_name=setting['test']['padder'].get('name', "pretrain"),
                                        device=device, padder_params=setting['test']['padder']['params'])
            test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=test_padder,
                                                **setting['test']['dataloader'])
            
            if setting['test'].get("check_teacher",False):
                print("\nTest Teacher model:")
                model = traj_clip
            else: 
                print("\nTest Trajectory-Mamba:")
                model = trajectory_mamba_model
            
            down_task = setting['test'].get('task', "destination_prediction")

            if down_task == "destination_prediction":
                if_denormalize = False
                if sorted(test_padder.pred_cols) == sorted([Y_COL, X_COL]): # 预测GPS
                    if_denormalize = True
                predictions, targets = test_model(model=model, pred_head=pred_head, dataloader=test_dataloader, denormalize=if_denormalize)
                print(f"{SAVE_NAME} predictions: {predictions.shape}")
                
                if test_padder.pred_cols == [ROAD_COL]: # 预测路段，分类任务
                    cal_metric = utils.cal_classification_metric
                    metric_filename = "road_classification"
                elif sorted(test_padder.pred_cols) == sorted([Y_COL, X_COL]): # 预测GPS
                    lng_col, lat_col = test_padder.pred_cols.index(X_COL), test_padder.pred_cols.index(Y_COL)
                    cal_metric = partial(utils.cal_distance_metric, lng_col=lng_col, lat_col=lat_col)
                    metric_filename = "gps_regression"
                elif test_padder.pred_cols == [DT_COL]: # 预测Arrival Time
                    cal_metric = utils.cal_regression_metric
                    metric_filename = "delta_t_regression"
                else:
                    raise NotImplementedError(f'No predict columns called "{test_padder.pred_cols}".')
                metric = cal_metric(targets, predictions)
                print(f"the test metric for {test_padder.pred_cols}:")
                print(metric)

            elif down_task == "search":
                eval_dataset = os.path.basename(setting['dataset']['test_traj_df']).split(".")[0]
                search_meta_dir = os.path.join(SEARCH_META_DIR, eval_dataset)
                try:
                    alltrajtgt, hopqrytgt, neg_indices = load_trajSearch_testdata(search_meta_dir, **setting['test']["search_data_params"])
                except FileNotFoundError:
                    print("No meta for similar trajectory search, please use data.py to Generate!")
                
                alltrajtgt_dataset = TrajectorySearchDataset(alltrajtgt)
                trajqrytgt_dataset = TrajectorySearchDataset(hopqrytgt)
                alltrajtgt_dataloader = DataLoader(alltrajtgt_dataset, shuffle=False, collate_fn=SearchPadder(device=device),
                                            **setting['test']['dataloader'])
                trajqrytgt_dataloader = DataLoader(trajqrytgt_dataset, shuffle=False, collate_fn=SearchPadder(device=device),
                                            **setting['test']['dataloader'])
                
                predictions, targets = test_model_on_search(model=model,
                                                            traj_dataloader=test_dataloader, # 从原始轨迹中选neg
                                                            qrytgt_dataloader=trajqrytgt_dataloader, neg_indices=neg_indices, test_embed_time=setting['test'].get('test_embed_time', False))
                metric_filename = "similar_trajectory_search"
                metric = utils.cal_classification_metric(targets, predictions)
                metric["mean_rank"] = utils.cal_mean_rank(predictions, targets)
                print(f"the test metric for similar trajectory search:")
                print(metric)
            
            else:
                raise NotImplementedError(f'No downstream task called "{down_task}".')

            if setting['test'].get('save', False):
                utils.create_if_noexists(os.path.join(PRED_SAVE_DIR, SAVE_NAME))
                np.save(os.path.join(PRED_SAVE_DIR, SAVE_NAME, 'predictions.npy'), predictions)
                np.save(os.path.join(PRED_SAVE_DIR, SAVE_NAME, 'targets.npy'), targets)
                metric.to_hdf(os.path.join(PRED_SAVE_DIR, SAVE_NAME, f'{metric_filename}.h5'), key='metric', format='table')


if __name__ == '__main__':
    main()
