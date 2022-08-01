import copy
import pandas as pd
from operator import itemgetter
import numpy as np
import random
from sklearn.cluster import k_means
import torch as th
import torch.nn as nn
import dgl
from dgl.data.utils import load_graphs
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
from ogb.nodeproppred import Evaluator
from math import sqrt


# convert the inputs from cpu to gpu, accelerate the running speed
def convert_to_gpu(*data, device: str):
    # overriding device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def set_random_seed(seed: int = 0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.random.seed(0)


def load_model(model: nn.Module, model_path: str):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def get_n_params(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: model
    :return: int
    """
    return sum(p.numel() for p in model.parameters())


def load_dataset(data_path: str, predict_category: str, data_split_idx_path: str = None):
    """
    load dataset
    :param data_path: data file path
    :param predict_category: predict node category
    :param data_split_idx_path: split index file path
    :return:
    """
    graph_list, labels = load_graphs(data_path)

    graph = graph_list[0]

    labels = labels[predict_category].squeeze(dim=-1)

    num_classes = len(labels.unique())

    split_idx = torch.load(data_split_idx_path)
    train_idx, valid_idx, test_idx = split_idx['train'][predict_category], split_idx['valid'][predict_category], split_idx['test'][predict_category]

    return graph, labels, num_classes, train_idx, valid_idx, test_idx


def get_predict_edge_index(graph: dgl.DGLGraph, sampled_edge_type: str or tuple,
                           sample_edge_rate: float, seed: int = 0):
    """
    get predict edge index, return train_edge_idx, valid_edge_idx, test_edge_idx
    :return:
    """

    torch.manual_seed(seed=seed)

    selected_edges_num = int(graph.number_of_edges(sampled_edge_type) * sample_edge_rate)
    permute_idx = torch.randperm(graph.number_of_edges(sampled_edge_type))

    train_edge_idx = permute_idx[: 3 * selected_edges_num]
    valid_edge_idx = permute_idx[3 * selected_edges_num: 4 * selected_edges_num]
    test_edge_idx = permute_idx[4 * selected_edges_num: 5 * selected_edges_num]

    return train_edge_idx, valid_edge_idx, test_edge_idx


def get_edge_data_loader(node_neighbors_min_num: int, n_layers: int,
                         graph: dgl.DGLGraph, batch_size: int, sampled_edge_type: str,
                         negative_sample_edge_num: int,
                         train_edge_idx: torch.Tensor, valid_edge_idx: torch.Tensor,
                         test_edge_idx: torch.Tensor,
                         reverse_etypes: dict, shuffle: bool = True, drop_last: bool = False,
                         num_workers: int = 4):
    """
    get edge data loader for link prediction, including train_loader, val_loader and test_loader
    :return:
    """
    # list of neighbors to sample per edge type for each GNN layer
    sample_nodes_num = []
    for layer in range(n_layers):
        sample_nodes_num.append({etype: node_neighbors_min_num + layer for etype in graph.canonical_etypes})

    # pos_sampler
    pos_sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_nodes_num)

    # neg sampler
    train_neg_sampler = dgl.dataloading.negative_sampler.Uniform(negative_sample_edge_num)

    # train_loader
    train_sampler = dgl.dataloading.as_edge_prediction_sampler(pos_sampler, exclude='reverse_types',reverse_etypes=reverse_etypes, negative_sampler=train_neg_sampler)
    train_loader = dgl.dataloading.DataLoader(graph, {sampled_edge_type: train_edge_idx}, train_sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    # val_loader
    val_neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    val_sampler = dgl.dataloading.as_edge_prediction_sampler(pos_sampler, exclude='reverse_types',reverse_etypes=reverse_etypes, negative_sampler=val_neg_sampler)
    val_loader = dgl.dataloading.DataLoader(graph, {sampled_edge_type: valid_edge_idx}, val_sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    # test_loader
    test_loader = dgl.dataloading.DataLoader(graph, {sampled_edge_type: test_edge_idx}, val_sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_optimizer_and_lr_scheduler(model: nn.Module, optimizer_name: str, learning_rate: float, weight_deacy: float, steps_per_epoch: int, epochs: int):
    """
    get optimizer and lr scheduler
    :param model:
    :param optimizer_name:
    :param learning_rate:
    :param weight_deacy:
    :param steps_per_epoch:
    :param epochs:
    :return:
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    else:
        raise ValueError(f"wrong value for optimizer {optimizer_name}!")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=learning_rate / 100)

    return optimizer, scheduler


def evaluate_link_prediction(predict_scores: torch.Tensor, true_scores: torch.Tensor):
    """
    get evaluation metrics for link prediction
    :param predict_scores: Tensor, shape (N, )
    :param true_scores: Tensor, shape (N, )
     :return: RMSE, MAE, AUC, AP to evaluate model performance in link prediction
    """
    RMSE = sqrt(mean_squared_error(true_scores.cpu().numpy(), predict_scores.cpu().numpy()))
    MAE = mean_absolute_error(true_scores.cpu().numpy(), predict_scores.cpu().numpy())
    AUC = roc_auc_score(true_scores.cpu().numpy(), predict_scores.cpu().numpy())
    AP = average_precision_score(true_scores.cpu().numpy(), predict_scores.cpu().numpy())

    return RMSE, MAE, AUC, AP


def evaluate_link_recommendation(predict_scores: torch.Tensor, true_scores: torch.Tensor, dst_node_representations: torch.Tensor, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor, K=10):
    """
    get evaluation metrics for link prediction
    :param predict_scores: Tensor, shape (N, )
    :param true_scores: Tensor, shape (N, )
    :return: 
    """
    ranked_info={
        'user':list(),
        'target':list(),
        'score':list(),
        'feat':list(),
        'true':list(),
    }
    for i, pred in sorted(enumerate(predict_scores), key=itemgetter(1), reverse=True):
        ranked_info['user'].append(src_node_ids[i].item())
        ranked_info['target'].append(dst_node_ids[i].item())
        ranked_info['score'].append(pred.item())
        ranked_info['feat'].append(dst_node_representations[i])
        ranked_info['true'].append(int(true_scores[i].item()))

    HIT=get_HIT_score(ranked_info,K)
    DIV=get_DIV_score(ranked_info,K)
    COV=get_COV_score(ranked_info,K)

    return HIT, DIV, COV

def get_user_HITscore(rankedUserInfo):
    user_HITs=0
    for i, val in rankedUserInfo['true'].items():
        if val==1:
            user_HITs+=1

    return user_HITs/len(rankedUserInfo['true'])

def get_HIT_score(ranked_info,K):
    user_list= np.unique(ranked_info['user'])
    total_hit_score=0
    for user in user_list:
        rankedUserInfo=get_user_ranked_info(user, ranked_info, K)
        total_hit_score+=get_user_HITscore(rankedUserInfo)
    return total_hit_score/len(user_list)


def get_user_DIVscore(rankedUserInfo):
    cosi = torch.nn.CosineSimilarity(dim=0)
    i_sum=0
    for i_index, i_emb in rankedUserInfo['feat'].items():
        j_sum=0
        for j_index, j_emb in rankedUserInfo['feat'].items():
            if j_index!=i_index:
                dissimilarity=1-cosi(i_emb,j_emb)
                j_sum+=dissimilarity
        i_sum+=j_sum
    n=len(rankedUserInfo['user'])
    factorizer=2/(n*(n-1))
    return factorizer*i_sum

def get_DIV_score(ranked_info,K):
    user_list= np.unique(ranked_info['user'])
    total_div_score=0
    for user in user_list:
        rankedUserInfo=get_user_ranked_info(user, ranked_info, K)
        total_div_score+=get_user_DIVscore(rankedUserInfo)
    return total_div_score/len(user_list)

def get_user_COVtargets(rankedUserInfo):
    user_target_ids=[]
    for _, target in rankedUserInfo['target'].items():
        user_target_ids.append(target)
    return user_target_ids

def get_COV_score(ranked_info,K):
    user_list= np.unique(ranked_info['user'])
    found_targets=[]
    for user in user_list:
        rankedUserInfo=get_user_ranked_info(user, ranked_info, K)
        found_targets+=get_user_COVtargets(rankedUserInfo)
    all_target_ids = np.unique([ranked_info['target'][i] for i, row_true in enumerate(ranked_info['true']) if row_true==1])
    matching_ids=0
    found_targets=np.unique(found_targets)
    for id in found_targets:
        if id in all_target_ids:
            matching_ids+=1
    return matching_ids/len(all_target_ids)



def get_user_ranked_info(user, ranked_info, K):
    #filter for a specific user
    user_ranked_info = {k: [x for i, x in enumerate(v) if ranked_info['user'][i] == user] for k, v in ranked_info.items()}
    # sort dataframe by score 
    user_ranked_info = pd.DataFrame(user_ranked_info).sort_values(by='score', ascending=False).drop_duplicates(subset='target').reset_index(drop = True)
    # splice by K
    # print(user_ranked_info.head(K))
    return user_ranked_info.head(K).to_dict()

    

