import copy
import json
import os
from tqdm import tqdm
import pandas as pd
from DGL_LFM1b.DGL_LFM1b import LFM1b
from dgl.data.utils import load_graphs
from model.R_HGNN import R_HGNN
import torch.nn as nn
from utils.utils import convert_to_gpu
import torch as th 
from utils.LinkScorePredictor import LinkScorePredictor
from DGL_LFM1b.data_utils import get_fileSize, get_col_names, setType, isValid, get_preprocessed_ids


def get_id_mapping(path_to_file, type, reverse=False):
    ids=get_preprocessed_ids(path_to_file, return_unique_ids=False, type=type, id_list=get_col_names(type))[f'{type}_id']
    if reverse:
        return {i: row for i, row in enumerate(ids)}
    else:
        return {row: i for i, row in enumerate(ids)}

def get_result_folder_path(root, date, sample_edge_type):
    return f'{root}/lfm1b/{date}/{sample_edge_type}'
    
def get_result_folder_args(root, date, sample_edge_type):
    return get_result_folder_path(root, date, sample_edge_type)+'/args.json'

def get_result_folder_model_state(root, date, sample_edge_type):
    return get_result_folder_path(root, date, sample_edge_type)+f'/{sample_edge_type}.pkl'

def build_model(graph,  date, args, sample_edge_type, root='results/'):
    model_state_path=get_result_folder_model_state(root, date, sample_edge_type)
    r_hgnn = R_HGNN(graph=graph,
                input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                hidden_dim=args['hidden_dim'], 
                relation_input_dim=args['rel_input_dim'],
                relation_hidden_dim=args['rel_hidden_dim'],
                num_layers=args['num_layers'], 
                n_heads=args['num_heads'], 
                dropout=args['dropout'],
                residual=args['residual'], 
                norm=args['norm'])
    link_scorer = LinkScorePredictor(args['hidden_dim'] * args['num_heads'])

    model = nn.Sequential(r_hgnn, link_scorer)
    model = convert_to_gpu(model, device=args['device'])
    model.load_state_dict(th.load(model_state_path, map_location=args['device']))
    return model

def get_user_recommendations(graph, date, sampled_edge_type, results_root_path, user, K=10):
    '''
    Description:
    the function get_user_recommendations takes the necessary parameters to preform TopK recommendation for a single user, and single edge type in the given graph

    Parameters:
    graph - a DGL LFM-1b HeteroGraph  
    data_pre_path - a string path to the preprocessed data files of the LFM-1b data loader
    date - is a string date and time used to identify the folder in which the compiled link prediciton models will be selected from
    sampled_edge_type - string indicator of the type of link prediction model to be used for TopK recommendation
    results_root_path - string path to the LFM-1b results directory
    user -  is the integer user id representing a single user from the LFM-1b data set
    K - is the number of recommmendations that will be made for this particular user (k=10 by default)

    Returns:
    A dictionaru of recommendations in the following form...

    result={
        'type_id':[artist_id #123,artist_id #213,artist_id #141,artist_id #223,......],
        'score:[.94,.84,.82,.75,.......]
    }
    '''
    args_path=get_result_folder_args(results_root_path, date, sampled_edge_type)
    args = json.load(open(args_path))
    type=sampled_edge_type.split('_')[-1]

    model=build_model(graph,  date, args, sampled_edge_type, root='./results/')
    input_features = {(stype, etype, dtype): graph.srcnodes[dtype].data['feat'] for stype, etype, dtype in graph.canonical_etypes}
    nodes_representation, _ = model[0].inference(graph, copy.deepcopy(input_features), device=args['device'])
    del model
    user_nodes_representation=nodes_representation['user']
    type_nodes_representation=nodes_representation[type]
    # C = torch.mm(A, B.T)  # same as C = A @ B.T
    listen_to_type_likelihood = th.mm(user_nodes_representation, type_nodes_representation.T)
    user_type_recommendations={}
    for u_id, row in enumerate(listen_to_type_likelihood):
        if u_id==user:
            for id, _ in enumerate(row):
                try:
                    graph.edge_id(u_id,id, etype=sampled_edge_type)
                except:
                    user_type_recommendations[id]=listen_to_type_likelihood[u_id,id].item()
    user_type_recommendations=sorted(user_type_recommendations.items(), key=lambda x:x[1], reverse=True)
    return dict(list(user_type_recommendations)[:K])
    
