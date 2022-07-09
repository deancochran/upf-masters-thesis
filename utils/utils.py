import copy
from operator import itemgetter
import numpy as np
import random
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
    print('splitting by users')
    selected_edges_num = int(graph.num_nodes('user') * sample_edge_rate)
    permute_idx = torch.randperm(graph.num_nodes('user'))
    user_train_edge_idx = permute_idx[: 3 * selected_edges_num]
    user_valid_edge_idx = permute_idx[3 * selected_edges_num: 4 * selected_edges_num]
    user_test_edge_idx = permute_idx[4 * selected_edges_num: 5 * selected_edges_num]
    src_dst_edges=graph.find_edges(torch.arange(graph.number_of_edges(sampled_edge_type), dtype=torch.int64), sampled_edge_type)
    train_edge_idx=[]
    for user in user_train_edge_idx:
        train_edge_idx+=[i for i,src in enumerate(src_dst_edges[0]) if user==src]
    valid_edge_idx=[]
    for user in user_valid_edge_idx:
        valid_edge_idx+=[i for i,src in enumerate(src_dst_edges[0]) if user==src]
    test_edge_idx=[]
    for user in user_test_edge_idx:
        test_edge_idx+=[i for i,src in enumerate(src_dst_edges[0]) if user==src]

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


def evaluate_link_recommendation(predict_scores: torch.Tensor, true_scores: torch.Tensor, dst_node_representations: torch.Tensor, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor, K=25):
    """
    get evaluation metrics for link prediction
    :param predict_scores: Tensor, shape (N, )
    :param true_scores: Tensor, shape (N, )
    :return: 
    """
    # print('predict_scores',predict_scores.shape)
    # print('true_scores',true_scores.shape) 
    # print('dst_node_representations',dst_node_representations.shape)
    # print('src_node_ids',src_node_ids.shape)
    # print('dst_node_ids',dst_node_ids.shape)

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
    # CONF=get_CONF_score(ranked_info,K)
    # COV=get_COV_score(ranked_info,K)
    # NOV=get_NOV_score()

    return HIT, DIV

def get_user_HITscore(user, ranked_info,K):
    user_HITs=0
    total=0
    index=0
    while total!=K and index < len(ranked_info['user']):
        row_user=ranked_info['user'][index]
        row_true=ranked_info['true'][index]
        if user==row_user:
            # print(row_user,row_score,row_true)
            if row_true==1:
                user_HITs+=1
            total+=1
        index+=1
    # print(f'hit @ {K} for user {user}',user_HITs/total)
    return user_HITs/total

def get_HIT_score(ranked_info,K):
    user_list= np.unique(ranked_info['user'])
    total_hit_score=0
    for user in user_list:
        if ranked_info['user'].count(user) >= K:
            total_hit_score+=get_user_HITscore(user, ranked_info, K)
    # print(f'HIT @ {K} for ALL users',total_hit_score/len(user_list))
    return total_hit_score/len(user_list)


def get_user_DIVscore(user, ranked_info,K):
    cosi = torch.nn.CosineSimilarity(dim=0)
    list_indicies=[]
    total=0
    index=0
    while total!=K and index < len(ranked_info['user']):
        row_user=ranked_info['user'][index]
        if user==row_user:
            # print(row_user,row_score,row_true)
            list_indicies.append(index)
            total+=1
        index+=1
    i_sum=0
    for i in list_indicies:
        j_sum=0
        for j in [val for val in list_indicies if val!=i]:
            dissimilarity=1-cosi(ranked_info['feat'][i],ranked_info['feat'][j])
            j_sum+=dissimilarity
        i_sum+=j_sum
    n=len(list_indicies)
    factorizer=2/(n*(n-1))

    # print(f'hit @ {K} for user {user}',user_HITs/total)
    return factorizer*i_sum

def get_DIV_score(ranked_info,K):
    user_list= np.unique(ranked_info['user'])
    total_div_score=0
    for user in user_list:
        if ranked_info['user'].count(user) >= K:
            total_div_score+=get_user_DIVscore(user, ranked_info, K)
    # print(f'DIV @ {K} for ALL users',total_div_score/len(user_list))
    return total_div_score/len(user_list)



# def get_user_CONFscore(user, ranked_info,K):
#     user_confidences=0
#     total=0
#     index=0
#     while total!=K and index < len(ranked_info['user']):
#         row_user=ranked_info['user'][index]
#         row_score=ranked_info['score'][index]
#         if user==row_user:
#             # print(row_user,row_score,row_true)
#             user_confidences+=row_score
#             total+=1
#         index+=1
#     # print(f'hit @ {K} for user {user}',user_HITs/total)
#     return user_confidences/total

# def get_CONF_score(ranked_info,K):
#     user_list= np.unique(ranked_info['user'])
#     total_conf_score=0
#     for user in user_list:
#         if ranked_info['user'].count(user) >= K:
#             total_conf_score+=get_user_CONFscore(user, ranked_info, K)
#     print(f'CONF @ {K} for ALL users',total_conf_score/len(user_list))
#     return total_conf_score/len(user_list)



# def get_user_COVscore(user, ranked_info,K):
#     user_target_ids=[]
#     while index < len(ranked_info['user']):
#         row_user=ranked_info['user'][index]
#         row_target=ranked_info['target'][index]
#         row_true =ranked_info['true'][index]
#         row_true =ranked_info['true'][index]
#         if user==row_user:
#             if row_true==1:
#                 user_target_ids.append(row_target)
#         index+=1
#     print(user_target_ids)
#     return user_target_ids

# def get_COV_score(ranked_info,K):
#     user_list= np.unique(ranked_info['user'])
#     found_targets=[]
#     for user in user_list:
#         if ranked_info['user'].count(user) >= K:
#             found_targets+=get_user_COVscore(user, ranked_info, K)
#     all_target_ids = [ranked_info['target'][i] for i, row_true in enumerate(ranked_info['true']) if row_true==1]
#     print('all_target_ids',all_target_ids)
#     print('found_targets',found_targets)
#     matching_ids=0
#     for id in found_targets:
#         if id in all_target_ids:
#             matching_ids+=1

#     print(f'COV @ {K} for ALL users',matching_ids/len(all_target_ids))
#     return matching_ids/len(all_target_ids)