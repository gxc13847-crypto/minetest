import os
import sys
import time
import copy
import math
import numpy as np
import torch
from tqdm import trange, tqdm
from models.trajectory_mamba import MEC
from data import TrajectorySearchTestdata, X_COL, Y_COL
from utils import cal_classification_metric, cal_distance_metric, cal_regression_metric, create_if_noexists


def _checkpoint_paths(checkpoint_dir, save_name):
    ckpt_path = os.path.join(checkpoint_dir, f'{save_name}_pretrain_checkpoint.pt')
    teacher_embeds_path = os.path.join(checkpoint_dir, f'{save_name}_teacher_embeds.npy')
    return ckpt_path, teacher_embeds_path


def _save_pretrain_checkpoint(checkpoint_dir, save_name, payload):
    create_if_noexists(checkpoint_dir)
    ckpt_path, teacher_embeds_path = _checkpoint_paths(checkpoint_dir, save_name)
    teacher_embeds = payload.pop('teacher_embeds', None)
    if teacher_embeds is not None:
        np.save(teacher_embeds_path, teacher_embeds)
        payload['teacher_embeds_path'] = teacher_embeds_path
    torch.save(payload, ckpt_path)
    print(f"Saved pretrain checkpoint: stage={payload['stage']}, epoch={payload['epoch']}, path={ckpt_path}")


def _load_pretrain_checkpoint(checkpoint_dir, save_name):
    ckpt_path, teacher_embeds_path = _checkpoint_paths(checkpoint_dir, save_name)
    if not os.path.exists(ckpt_path):
        return None
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    embeds_path = checkpoint.get('teacher_embeds_path', teacher_embeds_path)
    if os.path.exists(embeds_path):
        checkpoint['teacher_embeds'] = np.load(embeds_path)
    print(f"Loaded pretrain checkpoint: stage={checkpoint['stage']}, epoch={checkpoint['epoch']}, path={ckpt_path}")
    return checkpoint


def pretrain_model(model, teacher_model, dataset, dataloader, 
                   unshuffle_teacher_dataloader, teacher_dataloader,
                   num_epoch_1, num_epoch_2, lr_1, lr_2, MEC_weight=0, MEC_config=None,
                   teacher_prepared=False, teacher_savename=None,
                   checkpoint_dir=None, resume=False, save_name=None):
    """Pre-train the model with the given training dataloader, include two stages. 

    Args:
        model (nn.Module): the model to train.
        dataloader (DataLoader): batch iterator containing the training data.
        num_epoch (int): number of epoches to train.
        lr (float): learning rate for the optimizer.
    """

    num_iter = len(unshuffle_teacher_dataloader)
    batch_size = unshuffle_teacher_dataloader.batch_size
    if MEC_weight:
        MEC_loss = MEC(model.output_size, num_epoch_2, num_iter, batch_size, 
                            **MEC_config).to(model.device)
    else:
        MEC_loss = None
        print("No use MEC Loss.")

    checkpoint = None
    if checkpoint_dir and resume and save_name:
        checkpoint = _load_pretrain_checkpoint(checkpoint_dir, save_name)
        if checkpoint is not None:
            teacher_model.load_state_dict(checkpoint['teacher_model'])
            model.load_state_dict(checkpoint['model'])
            teacher_prepared = checkpoint.get('teacher_prepared', teacher_prepared)

    start_stage = 1
    start_epoch_1 = 0
    start_epoch_2 = 0
    teacher_embeds = None
    if checkpoint is not None:
        start_stage = checkpoint['stage']
        if start_stage == 1:
            start_epoch_1 = checkpoint['epoch'] + 1
        else:
            start_epoch_2 = 0 if checkpoint['epoch'] < 0 else checkpoint['epoch'] + 1
            teacher_embeds = checkpoint.get('teacher_embeds')

    pretrain_time_1 = None
    if not teacher_prepared and start_stage == 1:
        optimizer_t = torch.optim.Adam(teacher_model.parameters(), lr=lr_1)
        if checkpoint is not None and checkpoint.get('optimizer_t') is not None:
            optimizer_t.load_state_dict(checkpoint['optimizer_t'])
        teacher_model.train()

        bar_desc = 'Pretraining stage 1, avg loss: %.5f'
        whole_process_time = []
        start_time = time.time()
        with trange(start_epoch_1, num_epoch_1, initial=start_epoch_1, total=num_epoch_1,
                    desc=bar_desc % 0.0, position=0) as bar:
            for epoch_i in bar:
                loss_values = []
                traj_process_time = []
                for batch in tqdm(teacher_dataloader, desc='-->Traversing', leave=False, ncols=60):
                    optimizer_t.zero_grad()
                    loss, process_time = teacher_model.loss(*batch)
                    loss.backward()
                    optimizer_t.step()
                    loss_values.append(loss.item())
                    traj_process_time.append(process_time)
                bar.set_description(bar_desc % np.mean(loss_values))
                whole_process_time.append(np.sum(traj_process_time))
                if checkpoint_dir and save_name:
                    _save_pretrain_checkpoint(checkpoint_dir, save_name, {
                        'stage': 1,
                        'epoch': epoch_i,
                        'teacher_prepared': False,
                        'teacher_model': teacher_model.state_dict(),
                        'model': model.state_dict(),
                        'optimizer_t': optimizer_t.state_dict(),
                        'optimizer': None,
                        'mec_loss': None,
                    })
        end_time = time.time()
        trained_epochs_1 = max(num_epoch_1 - start_epoch_1, 1)
        pretrain_time_1 = ((end_time-start_time)-np.sum(whole_process_time)) / (trained_epochs_1*60)
        print(f"Stage 1's mean pretrain time: {pretrain_time_1} min/epoch")
        if teacher_savename is not None:
            create_if_noexists(os.path.dirname(teacher_savename) or '.')
            torch.save(teacher_model.state_dict(), teacher_savename)
        teacher_prepared = True

    if start_stage == 1 or checkpoint is None:
        model.traj_view.load_state_dict(copy.deepcopy(teacher_model.traj_view).state_dict())

    if MEC_weight:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(MEC_loss.parameters()), lr=lr_2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_2)
    if checkpoint is not None and checkpoint.get('optimizer') is not None and start_stage >= 2:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if MEC_loss is not None and checkpoint.get('mec_loss') is not None:
            MEC_loss.load_state_dict(checkpoint['mec_loss'])
    teacher_model.eval()
    model.train()
    
    bar_desc = 'Pretraining stage 2, avg loss: %.5f, average compressed ratio: %.3f%%'
    whole_process_time = []
    start_time = time.time()
    
    num_traj = len(unshuffle_teacher_dataloader.dataset)
    if teacher_embeds is not None and len(teacher_embeds) != num_traj:
        print(
            f"teacher_embeds length ({len(teacher_embeds)}) != dataset ({num_traj}); "
            "regenerating teacher embeds."
        )
        teacher_embeds = None

    if teacher_embeds is None:
        teacher_embeds = []
        traj_process_time = []
        with torch.no_grad():
            for batch in tqdm(unshuffle_teacher_dataloader, desc='Generating teacher embeds', ncols=60):
                teacher_traj_h, _, process_time  = teacher_model.forward_w_processtime_cal(*batch)
                teacher_embeds.append(teacher_traj_h.cpu().numpy())
                traj_process_time.append(process_time)
        teacher_embeds = np.concatenate(teacher_embeds, 0)
        whole_process_time.append(np.sum(traj_process_time))
        if checkpoint_dir and save_name:
            _save_pretrain_checkpoint(checkpoint_dir, save_name, {
                'stage': 2,
                'epoch': -1,
                'teacher_prepared': True,
                'teacher_model': teacher_model.state_dict(),
                'model': model.state_dict(),
                'optimizer_t': None,
                'optimizer': optimizer.state_dict(),
                'mec_loss': MEC_loss.state_dict() if MEC_loss is not None else None,
                'teacher_embeds': teacher_embeds,
            })

    dataset = dataset(teacher_embeds=teacher_embeds)
    dataloader = dataloader(dataset=dataset)
    with trange(start_epoch_2, num_epoch_2, initial=start_epoch_2, total=num_epoch_2,
                desc=bar_desc % (0.0, 100.0), position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            traj_process_time = []
            compressed_ratios = []
            for batch_i, batch in enumerate(tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60)):
                it = num_iter * epoch_i + batch_i
                
                optimizer.zero_grad()
                loss, compressed_valid_len, process_time = model.loss(*batch[0], batch[1], MEC_weight, MEC_loss, it)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
                traj_process_time.append(process_time)
                compressed_ratios.append(compressed_valid_len / batch[0][1])
            
            compressed_ratios = torch.cat(compressed_ratios, 0)
            bar.set_description(bar_desc % (np.mean(loss_values), compressed_ratios.mean().item() * 100))
            whole_process_time.append(np.sum(traj_process_time))
            if checkpoint_dir and save_name:
                _save_pretrain_checkpoint(checkpoint_dir, save_name, {
                    'stage': 2,
                    'epoch': epoch_i,
                    'teacher_prepared': True,
                    'teacher_model': teacher_model.state_dict(),
                    'model': model.state_dict(),
                    'optimizer_t': None,
                    'optimizer': optimizer.state_dict(),
                    'mec_loss': MEC_loss.state_dict() if MEC_loss is not None else None,
                    'teacher_embeds': teacher_embeds,
                })
    
    end_time = time.time()
    trained_epochs_2 = max(num_epoch_2 - start_epoch_2, 1)
    pretrain_time_2 = ((end_time-start_time)-np.sum(whole_process_time)) / (trained_epochs_2*60)
    print(f"Stage 2's mean pretrain time: {pretrain_time_2} min/epoch")
    if pretrain_time_1 is not None:
        print(f"Mean pretrain time: {(pretrain_time_1*num_epoch_1 + pretrain_time_2*num_epoch_2)/(num_epoch_1+num_epoch_2)} min/epoch")


def finetune_model(model, pred_head, dataloader, num_epoch, lr, ft_encoder=True, weight_decay=0., clip=None, 
                   val_dataloader=None, valid_metrics_list=[], patience=10, 
                   denormalize=False, pred_cols=None):
    """Fine-tune the model with specific task labels.

    Args:
        model (nn.Module): the model to finetune.
        pred_head (nn.Module): the prediction head for mapping the embeddings to predictions.
        dataloader (DataLoader): batch iterator containing the finetune data.
        num_epoch (int): number of epoches to finetune.
        lr (float): learning rate of the optimizer.
        ft_encoder (bool, optional): Whether to finetune the trajectory encoder. Defaults to True.
        If set to False, then only the task-specific prediction module will be finetuned.
    """
    pred_head.train()
    if ft_encoder:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(pred_head.parameters()), lr=lr, weight_decay=weight_decay)
        model.train()
    else:
        optimizer = torch.optim.Adam(pred_head.parameters(), lr=lr, weight_decay=weight_decay)
        model.eval()

    best_model_state, best_pred_head_state = None, None
    wait = 0
    min_eval_metric = float("inf")
    bar_desc = 'Finetuning, avg loss: %.5f'
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            # norm_grads = []
            # norm_grads_m = []
            for batch in tqdm(dataloader, desc='-->Traversing', leave=False, ncols=60):
                *input_batch, label = batch

                optimizer.zero_grad()
                traj_h = model(*input_batch)
                if not ft_encoder:
                    traj_h = traj_h.detach()
                loss = pred_head.loss(traj_h, label, denormalize)
                loss.backward()
                # # check梯度范数（调试用）
                # norm_grad = torch.nn.utils.clip_grad_norm_(pred_head.parameters(), float('inf'))
                # norm_grad_m = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                # norm_grads.append(norm_grad.item())
                # norm_grads_m.append(norm_grad_m.item())

                if clip is not None:
                    if ft_encoder: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    torch.nn.utils.clip_grad_norm_(pred_head.parameters(), clip)
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))
            # print(f"Grad Norm: {np.mean(norm_grads):.2e}")
            # print(f"Grad Norm: {np.mean(norm_grads_m):.2e}")

            def cal_valid_metric():
                if pred_head.pred_type == 'regression':
                    if denormalize:
                        lng_col, lat_col = pred_cols.index(X_COL), pred_cols.index(Y_COL)
                        valid_metrics = cal_distance_metric(targets, predictions, lng_col, lat_col)
                        eval_metric = valid_metrics['distance_mae']
                    else:
                        valid_metrics = cal_regression_metric(targets, predictions)
                        eval_metric = valid_metrics['mae']
                elif pred_head.pred_type == 'classification':
                    valid_metrics = cal_classification_metric(targets, predictions)
                    eval_metric = -valid_metrics['acc@1']
                else:
                    raise NotImplementedError(f'No prediction type: {pred_head.pred_type}.')
                return eval_metric, valid_metrics
            
            if val_dataloader is not None:
                predictions, targets =  test_model(model, pred_head, val_dataloader, denormalize) # np.array
                pred_head.train()
                if ft_encoder: model.train()
                eval_metric, valid_metrics = cal_valid_metric()
                print(f"eval_metric={eval_metric}.")
                valid_metrics_list.append(valid_metrics)
                if eval_metric < min_eval_metric: # 越小越好
                    wait = 0
                    min_eval_metric = eval_metric
                    best_model_state = copy.deepcopy(model).state_dict()
                    best_pred_head_state = copy.deepcopy(pred_head).state_dict()
                    best_epoch = epoch_i
                else:
                    wait+=1
                    if wait>=patience:
                        print(f"Early Stop at epoch {epoch_i+1}.")
                        break

    if best_model_state is not None:
        print(f"Best models are at epoch {best_epoch+1}, load their state.")
        model.load_state_dict(best_model_state)
        pred_head.load_state_dict(best_pred_head_state)

@torch.no_grad()
def test_model(model, pred_head, dataloader, denormalize=False):
    """Test the model with specific prediction tasks.

    Args:
        model (nn.Module): the trajectory embedding model to test.
        pred_head (nn.Module): the prediction head for mapping trajectory embeddings to predictions.
        dataloader (DataLoader): batch iterator containing the testing data.
    """
    model.eval()
    pred_head.eval()

    predictions, targets = [], []
    for batch in tqdm(dataloader, 'Testing', ncols=60):
        *input_batch, target = batch
        traj_h = model(*input_batch)
        pred = pred_head(traj_h)
        if denormalize:
            pred = pred * (pred_head.spatial_border[1] - pred_head.spatial_border[0]).unsqueeze(0) + \
                        pred_head.spatial_border[0].unsqueeze(0)
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    predictions = np.concatenate(predictions, 0)
    targets = np.concatenate(targets, 0)
    return predictions, targets


@torch.no_grad()
def test_model_on_search(model, traj_dataloader, qrytgt_dataloader, neg_indices, test_embed_time=False, set_name="test"):
    """Test the model with similar trajectory search.

    Args:
        model (nn.Module): the trajectory embedding model to test.
        dataloader (DataLoader): batch iterator containing the testing data.
    """
    model.eval()

    qrytgt_embeds = []
    for batch_meta in tqdm(qrytgt_dataloader,
                            desc=f"Calculating query and target embeds on {set_name} set",
                            total=len(qrytgt_dataloader), ncols=60):
        encodes = model(*batch_meta)
        qrytgt_embeds.append(encodes.detach().cpu().numpy())
    qrytgt_embeds = np.concatenate(qrytgt_embeds, 0)
    qry_indices, tgt_indices = TrajectorySearchTestdata.parse_label(len(qrytgt_embeds))

    if test_embed_time:
        loop_times = 20
        embed_time_list, process_time_list = [],[]
        with trange(loop_times, desc=f"Calculating embedding time with {loop_times} loops", position=0, ncols=60) as bar:
            for _ in bar:
                enc_time_list = []
                for batch_meta in traj_dataloader:
                    _, enc_time, process_time = model.forward_on_search_mode(*batch_meta)
                    enc_time_list.append(enc_time)
                    process_time_list.append(process_time)
                embed_time_list.append(np.sum(enc_time_list))
        embed_time_list = np.array(embed_time_list)
        process_time_list = np.array(process_time_list)
        print(f"Total embedding time of {loop_times} loops: {embed_time_list.sum()}s")
        print(f"Mean embedding time on {set_name} set: {embed_time_list.sum()/loop_times}s")
        print(f"Min embedding time on {set_name} set: {embed_time_list.min()}s")
        print("Check total traj process time: {:.3f}s".format(process_time_list.sum()))


    embeds = []
    whole_enc_time = []
    traj_process_time = []
    for batch_meta in tqdm(traj_dataloader,
                            desc=f"Calculating embeds on {set_name} set",
                            total=len(traj_dataloader), ncols=60):
        encodes, enc_time, process_time = model.forward_on_search_mode(*batch_meta)
        embeds.append(encodes.detach().cpu().numpy())
        whole_enc_time.append(enc_time)
        traj_process_time.append(process_time)
    whole_enc_time = np.array(whole_enc_time)
    traj_process_time = np.array(traj_process_time)
    print("Embedding time: {:.3f}s".format(whole_enc_time.sum()))
    print("Check traj process time: {:.3f}s".format(traj_process_time.sum()))
    embeds = np.concatenate(embeds, 0)

    predictions, targets = TrajectorySearchTestdata.cal_pres_and_labels(qrytgt_embeds[qry_indices], qrytgt_embeds[tgt_indices], embeds[neg_indices])

    return predictions, targets


def print_mamba_key_params(model, epoch=None):
    """
    打印Mamba层部分参数的范围，用于监控训练中是否发生异常漂移
    Args:
        model: 模型实例（包含 self.trend_func = Mamba(...)）
        epoch: 当前训练轮次（可选，用于标记输出）
    """
    if epoch is not None:
        print(f"\n===== 第 {epoch} 轮训练后，关键参数范围 =====")
    else:
        print("\n===== Mamba关键参数范围 =====")
    
    mamba = model.trend_func  # 获取Mamba层实例
    with torch.no_grad():  # 不计算梯度，仅打印
        # 1. 打印x_proj权重范围（影响C矩阵，需关注是否过大）
        x_proj_weight = mamba.x_proj.weight
        print(f"1. x_proj权重范围: "
              f"[{x_proj_weight.min().item():.4f}, {x_proj_weight.max().item():.4f}] "
              f"（正常应在[-1.0, 1.0]内）")
        
        # 2. 打印A_log及对应的A值范围（A = -exp(A_log)，需确保足够负）
        A_log = mamba.A_log
        A = -torch.exp(A_log.float())  # 还原A矩阵
        print(f"2. A_log范围: "
              f"[{A_log.min().item():.4f}, {A_log.max().item():.4f}]")
        print(f"   对应的A值（衰减项）范围: "
              f"[{A.min().item():.4f}, {A.max().item():.4f}] "
              f"（正常应≤-1.0，越小衰减越强）")
        
        # 3. 打印dt_proj的权重和偏置范围（影响dt大小，需避免极端值）
        dt_proj_weight = mamba.dt_proj.weight
        dt_proj_bias = mamba.dt_proj.bias
        print(f"3. dt_proj权重范围: "
              f"[{dt_proj_weight.min().item():.4f}, {dt_proj_weight.max().item():.4f}] "
              f"（正常应在[-1.0, 1.0]内）")
        print(f"   dt_proj偏置范围: "
              f"[{dt_proj_bias.min().item():.4f}, {dt_proj_bias.max().item():.4f}] "
              f"（影响dt大小，过负会导致dt→0）")
    
    print("======================================")