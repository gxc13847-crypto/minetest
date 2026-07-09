import os
import string
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, accuracy_score, roc_auc_score


def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 'D' + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)


def get_random_string(length):
    letters = string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def create_if_noexists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cal_courseAngle(a_coor, b_coor):
    # lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    # y = np.sin(lng2-lng1) * np.cos(lat2)
    # x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng2-lng1)
    a_coor, b_coor = np.radians(a_coor), np.radians(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x = b_x-a_x
    y = np.sin(d_x) * np.cos(b_y)
    x = np.cos(a_y) * np.sin(b_y) - np.sin(a_y) * np.cos(b_y) * np.cos(d_x)
    bearing = np.arctan2(y, x)
    bearing = 180 * bearing / np.pi
    bearing = np.where(bearing < 0, bearing + 360, bearing)
    return bearing

def cal_geo_distance(a_coor, b_coor):
    """ Calculcate the geographical distance between two points (or one target point and an array of points). """
    # lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    # dlon = lng2 - lng1
    # dlat = lat2 - lat1
    # a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    # distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
    # return distance
    a_coor, b_coor = np.radians(a_coor), np.radians(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x = a_x - b_x # lng
    d_y = a_y - b_y # lat

    a = np.sin(d_y / 2) ** 2 + np.cos(a_y) * np.cos(b_y) * np.sin(d_x / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
    return distance

def cal_tensor_geo_distance(a_coor:torch.tensor, b_coor:torch.tensor):
    """ Calculcate the geographical distance between two points (or one target point and an array of points). """
    # lng1, lat1, lng2, lat2 = map(torch.deg2rad, [lng1, lat1, lng2, lat2])
    # dlon = lng2 - lng1
    # dlat = lat2 - lat1
    # a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    # distance = 2 * torch.arcsin(torch.sqrt(a)) * 6371 * 1000 # + 1e-8——不能在a后加，出大问题坏！
    # return distance
    a_coor, b_coor = torch.deg2rad(a_coor), torch.deg2rad(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x = a_x - b_x
    d_y = a_y - b_y

    a = torch.sin(d_y / 2) ** 2 + torch.cos(a_y) * torch.cos(b_y) * torch.sin(d_x / 2) ** 2
    distance = 2 * torch.arcsin(torch.sqrt(a)) * 6371 * 1000
    return distance

# def geo_distance(a_coor, b_coor):
#     a_coor, b_coor = torch.deg2rad(a_coor), torch.deg2rad(b_coor)
#     a_x, a_y = a_coor[..., 0], a_coor[..., 1]
#     b_x, b_y = b_coor[..., 0], b_coor[..., 1]
#     d_x = a_x - b_x
#     d_y = a_y - b_y

#     a = torch.sin(d_y / 2) ** 2 + torch.cos(a_y) * torch.cos(b_y) * torch.sin(d_x / 2) ** 2
#     distance = 2 * torch.arcsin(torch.sqrt(a)) * 6371 * 1000
#     return distance

# no use!
def cal_tensor_courseAngle(lng1:torch.tensor, lat1:torch.tensor, lng2:torch.tensor, lat2:torch.tensor):
    lng1, lat1, lng2, lat2 = map(torch.deg2rad, [lng1, lat1, lng2, lat2])
    y = torch.sin(lng2-lng1) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(lng2-lng1)
    bearing = torch.arctan2(y, x)
    bearing = 180 * bearing / torch.pi
    bearing = torch.where(bearing < 0, bearing + 360, bearing)
    return bearing


def gen_causal_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()


def get_batch_mask(B, L, valid_len):
    # mask = repeat(torch.arange(end=L, device=valid_len.device),
    #               'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
    mask = torch.arange(end=L, device=valid_len.device).unsqueeze(0) >= valid_len.unsqueeze(-1)  # (B, L)
    return mask


def tokenize_timestamp(t):
    week = t[..., 0] % (7 * 24 * 60 * 60) / (24 * 60 * 60)
    hour = t[..., 0] % (24 * 60 * 60) / (60 * 60)
    minute = t[..., 0] % (60 * 60) / 60
    d_minute = t[..., 1] / 60
    timestamp = t[..., 0] / 60
    return week, hour, minute, d_minute, timestamp


def pack_input(hidden_states, mask):
    """        
    :param hidden_states: Shape is [B,L,H]
    :param mask: Shape is [B,L]
    :return: 
    """
    hidden_states = rearrange(hidden_states, "b s ... -> (b s) ...")
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    packed_hidden_states = hidden_states[indices]
    return packed_hidden_states, indices

def pad_input(hidden_states, indices, batch, seqlen):
    """        
    :param hidden_states: Shape is [B*L,H] not [B,L,H]
    :param indices: from unpad_input return indices
    :param batch: 
    :param seqlen: max seqlen in batch
    :return: 
    """
    output = torch.zeros(batch * seqlen, *hidden_states.shape[1:], device=hidden_states.device,dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, '(b s) ... -> b s ...', b=batch)


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value
    

def mean_absolute_percentage_error(y_true, y_pred):
    """ Calculcates the MAPE metric. """
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape

def cal_regression_metric(label, pres):
    """ Calculcate all common regression metrics. """
    rmse = math.sqrt(mean_squared_error(label, pres))
    mae = mean_absolute_error(label, pres)
    mape = mean_absolute_percentage_error(label, pres)

    s = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
    return s


def distance_mae(distance, null_val=np.nan):
    distance_mae = np.mean(np.abs(distance))
    return distance_mae

def distance_mse(distance, null_val=np.nan):
    distance_mse = np.mean(distance**2)
    return distance_mse

def distance_rmse(distance, null_val=np.nan):
    return np.sqrt(distance_mse(distance=distance, null_val=null_val))

def cal_distance_metric(label, pres, lng_col, lat_col):
    """ 
    Calculcate all distance regression metrics. 

    :param labels: longitude and latitude features of the trajectories, with shape (B,2).
    :param pres: predicted longitude and latitude features of the trajectories, with shape (B, 2).
    """
    distance = cal_geo_distance(label[...,[lng_col, lat_col]], pres[...,[lng_col, lat_col]])
    mae = distance_mae(distance, 0.0)#.item()
    rmse = distance_rmse(distance, 0.0)#.item()
    s = pd.Series([rmse, mae], index=['distance_rmse', 'distance_mae'])
    return s


def top_n_accuracy(truths, preds, n):
    """ Calculcate Acc@N metric. """
    # best_n = np.argsort(preds, axis=1)[:, -n:] # 升序排列求后n个
    best_n = np.argsort(-preds, axis=1)[:, :n] # 降序排列求前n个
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classification_metric(labels, pres):
    """
    Calculates all common classification metrics.

    :param labels: classification label, with shape (N).
    :param pres: predicted classification distribution, with shape (N, num_class).
    """
    pres_index = pres.argmax(-1)  # (N)
    macro_f1 = f1_score(labels, pres_index, average='macro', zero_division=0)
    macro_recall = recall_score(labels, pres_index, average='macro', zero_division=0)
    acc = accuracy_score(labels, pres_index)
    n_list = [5, 10]
    top_n_acc = [top_n_accuracy(labels, pres, n) for n in n_list]

    s = pd.Series([macro_f1, macro_recall, acc] + top_n_acc,
                  index=['macro_f1', 'macro_rec'] +
                  [f'acc@{n}' for n in [1] + n_list])
    return s


def cal_mean_rank(scores, target_indices):
    """
    Calculate the Mean Rank metric.

    :param scores: A 2D NumPy array where each row contains the predicted scores for each label.
    :param target_indices: A 1D NumPy array containing the index of the target item in each prediction.
    :return: The value of Mean Rank.
    """
    # Get the ranks of each score in descending order
    ranks = scores.argsort(axis=1)[:, ::-1].argsort(axis=1) + 1

    # Extract the ranks of the target indices
    target_indices = target_indices.astype(int)
    target_ranks = ranks[np.arange(len(target_indices)), target_indices]

    # # Cap ranks greater than 100 at 100
    # target_ranks[target_ranks > 100] = 100
    # # # Calculate mean rank only for ranks <= 100
    # # target_ranks = target_ranks[target_ranks <= 100]

    # Calculate the mean of these ranks
    mean_rank_value = np.mean(target_ranks)# if target_ranks.size > 0 else float('inf') # handle case where no ranks are <= 10
    return mean_rank_value

def cal_mean_reciprocal_rank(scores, target_indices):
    """
    Calculate the Mean Reciprocal Rank metric.

    :param scores: A 2D NumPy array where each row contains the predicted scores for each label.
    :param target_indices: A 1D NumPy array containing the index of the target item in each prediction.
    :return: The value of Mean Reciprocal Rank.
    """
    # Get the ranks of each score in descending order
    ranks = scores.argsort(axis=1)[:, ::-1].argsort(axis=1) + 1

    # Extract the ranks of the target indices
    target_indices = target_indices.astype(int)
    target_ranks = ranks[np.arange(len(target_indices)), target_indices]

    # Calculate the mean of these ranks' reciprocal
    mean_reciprocal_rank_value = np.mean(np.reciprocal(target_ranks))
    return mean_reciprocal_rank_value


def cal_search_metric(labels, pres):
    """
    Calculates all metrics for similar trajectory search.

    :param labels: classification label, with shape (N).
    :param pres: predicted classification distribution, with shape (N, num_class).
    """
    # s = cal_classification_metric(labels, pres)
    pres_index = pres.argmax(-1)  # (N)
    acc = accuracy_score(labels, pres_index)
    acc5 = top_n_accuracy(labels, pres, 5)
    mean_rank = cal_mean_rank(pres, labels)
    
    s = pd.Series([acc, acc5] + mean_rank,
                  index=[f'acc@{n}' for n in [1,5]] + ["mean_rank"])
    return s


def cal_model_size(model):
    """ Calculate the total learnable parameter size (in megabytes) of a torch module. """
    param_size = 0
    for param in model.parameters():
        if param.requires_grad:
            param_size += param.nelement() * param.element_size()

    size_all_mb = param_size / 1024**2
    return size_all_mb

def cal_models_size(models: list):
    """ Calculate the total learnable parameter size (in megabytes) of a list of torch modules. """
    param_size = 0
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                param_size += param.nelement() * param.element_size()

    size_all_mb = param_size / 1024**2
    return size_all_mb


def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def gather_all_param(*models):
    parameters = []
    for encoder in models:
        parameters += list(encoder.parameters())
    return parameters