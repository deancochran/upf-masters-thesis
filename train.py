import argparse
import copy
import json
import os
import shutil
import warnings
import numpy as np
import torch as th 
import torch.nn as nn
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
from DGL_LFM1b.DGL_LFM1b import LFM1b
from utils.LinkScorePredictor import LinkScorePredictor
from utils.EarlyStopping import EarlyStopping
from model.R_HGNN import R_HGNN
from utils.utils import *
th.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
def smooth(y, box_pts=3):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def make_loss_plots(sample_edge_type, total_loss_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(total_loss_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), smooth(total_loss_vals['train']+zero_pad), 'r', label='Training loss')
    plt.plot(range(epochs), smooth(total_loss_vals['val']+zero_pad), 'g', label='Val loss')
    plt.plot(range(epochs), smooth(total_loss_vals['test']+zero_pad), 'b', label='Test loss')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_loss_plot.png")

def make_AUC_plots(sample_edge_type, AUC_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(AUC_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), smooth(AUC_vals['train']+zero_pad), 'r', label='Training AUC')
    plt.plot(range(epochs), smooth(AUC_vals['val']+zero_pad), 'g', label='Val AUC')
    plt.plot(range(epochs), smooth(AUC_vals['test']+zero_pad), 'b', label='Test AUC')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_auc_plot.png")


def make_AP_plots(sample_edge_type, AP_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(AP_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), smooth(AP_vals['train']+zero_pad), 'r', label='Training AP')
    plt.plot(range(epochs), smooth(AP_vals['val']+zero_pad), 'g', label='Val AP')
    plt.plot(range(epochs), smooth(AP_vals['test']+zero_pad), 'b', label='Test AP')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test AP')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_ap_plot.png")


def make_RMSE_plots(sample_edge_type, RMSE_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(RMSE_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), smooth(RMSE_vals['train']+zero_pad), 'r', label='Training RMSE')
    plt.plot(range(epochs), smooth(RMSE_vals['val']+zero_pad), 'g', label='Val RMSE')
    plt.plot(range(epochs), smooth(RMSE_vals['test']+zero_pad), 'b', label='Test RMSE')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_rmse_plot.png")

def make_MAE_plots(sample_edge_type, MAE_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(MAE_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), smooth(MAE_vals['train']+zero_pad), 'r', label='Training MAE')
    plt.plot(range(epochs), smooth(MAE_vals['val']+zero_pad), 'g', label='Val MAE')
    plt.plot(range(epochs), smooth(MAE_vals['test']+zero_pad), 'b', label='Test MAE')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_mae_plot.png")

def make_HIT_plots(sample_edge_type, HIT_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(HIT_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), smooth(HIT_vals['train']+zero_pad), 'r', label='Training HIT')
    plt.plot(range(epochs), smooth(HIT_vals['val']+zero_pad), 'g', label='Val HIT')
    plt.plot(range(epochs), smooth(HIT_vals['test']+zero_pad), 'b', label='Test HIT')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test HIT@25')
    plt.xlabel('Epochs')
    plt.ylabel('HIT')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_hit_plot.png")

def make_DIV_plots(sample_edge_type, DIV_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(DIV_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), smooth(DIV_vals['train']+zero_pad), 'r', label='Training DIV')
    plt.plot(range(epochs), smooth(DIV_vals['val']+zero_pad), 'g', label='Val DIV')
    plt.plot(range(epochs), smooth(DIV_vals['test']+zero_pad), 'b', label='Test DIV')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test DIV@25')
    plt.xlabel('Epochs')
    plt.ylabel('DIV')
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_div_plot.png")

def evaluate(model, loader, loss_func, sampled_edge_type, device, mode):
    """
    :param model: model
    :param loader: data loader (validate or test)
    :param loss_func: loss function
    :param sampled_edge_type: str
    :param device: device str
    :param mode: str, evaluation mode, validate or test
    :return:
    total_loss, y_trues, y_predicts
    """
    model.eval()
    with th.no_grad():
        y_trues = []
        y_predicts = []
        y_dst_node_representations=[]
        y_src_node_ids=[]
        y_dst_node_ids=[]
        total_loss = 0.0
        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(loader):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(positive_graph, negative_graph, device=device)
            # target node relation representation in the heterogeneous graph
            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                              blocks[0].canonical_etypes}

            nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features))

            positive_score, pos_dst_node_feat, pos_node_src_id, pos_node_dst_id = model[1](positive_graph, nodes_representation, sampled_edge_type)
            positive_score=positive_score.squeeze(dim=-1)
            pos_dst_node_feat=pos_dst_node_feat.squeeze(dim=-1)
            pos_node_src_id=pos_node_src_id.squeeze(dim=-1)
            pos_node_dst_id=pos_node_dst_id.squeeze(dim=-1)

            negative_score, neg_dst_node_feat, neg_node_src_id, neg_node_dst_id = model[1](negative_graph, nodes_representation, sampled_edge_type)
            negative_score=negative_score.squeeze(dim=-1)
            neg_dst_node_feat=neg_dst_node_feat.squeeze(dim=-1)
            neg_node_src_id=neg_node_src_id.squeeze(dim=-1)
            neg_node_dst_id=neg_node_dst_id.squeeze(dim=-1)


            y_predict = th.cat([positive_score, negative_score], dim=0)
            y_true = th.cat([th.ones_like(positive_score), th.zeros_like(negative_score)], dim=0)
            y_dst_node_representation = th.cat([pos_dst_node_feat, neg_dst_node_feat], dim=0)
            y_src_node_id = th.cat([pos_node_src_id, neg_node_src_id], dim=0)
            y_dst_node_id = th.cat([pos_node_dst_id, neg_node_dst_id], dim=0)

            loss = loss_func(y_predict, y_true)

            total_loss += loss.item()
            y_trues.append(y_true.detach().cpu())
            y_predicts.append(y_predict.detach().cpu())
            y_dst_node_representations.append(y_dst_node_representation.detach().cpu())
            y_src_node_ids.append(y_src_node_id.detach().cpu())
            y_dst_node_ids.append(y_dst_node_id.detach().cpu())

        total_loss /= (batch + 1)
        y_trues = th.cat(y_trues, dim=0)
        y_predicts = th.cat(y_predicts, dim=0)
        y_dst_node_representations = th.cat(y_dst_node_representations, dim=0)
        y_src_node_ids = th.cat(y_src_node_ids, dim=0)
        y_dst_node_ids = th.cat(y_dst_node_ids, dim=0)

    return total_loss, y_trues, y_predicts, y_dst_node_representations, y_src_node_ids, y_dst_node_ids

def train_model(model, optimizer,scheduler, train_loader, val_loader, test_loader, save_folder, sampled_edge_type, date, args):
    
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_folder, save_model_name=sampled_edge_type)
    tqdm_loader = tqdm(range(args.epochs), total=args.epochs)
    loss_func = nn.BCELoss()
    train_steps = 0
    best_validate_RMSE, final_result = None, None

    total_loss_vals={'train':[],'val':[],'test':[]}
    RMSE_vals={'train':[],'val':[],'test':[]}
    MAE_vals={'train':[],'val':[],'test':[]}
    AUC_vals={'train':[],'val':[],'test':[]}
    AP_vals={'train':[],'val':[],'test':[]}
    HIT_vals={'train':[],'val':[],'test':[]}
    DIV_vals={'train':[],'val':[],'test':[]}
    for epoch in tqdm_loader:
        model.train()
        train_y_trues = []
        train_y_predicts = []
        train_y_dst_node_representations=[]
        train_y_src_node_ids=[]
        train_y_dst_node_ids=[]
        train_total_loss = 0.0
        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_loader):
            blocks = [convert_to_gpu(b, device=args.device) for b in blocks]
            blocks = [convert_to_gpu(b, device=args.device) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(positive_graph, negative_graph, device=args.device)

            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in blocks[0].canonical_etypes}

            nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features), args=args)

            positive_score, pos_dst_node_feat, pos_node_src_id, pos_node_dst_id = model[1](positive_graph, nodes_representation, sampled_edge_type)
            positive_score=positive_score.squeeze(dim=-1)
            pos_dst_node_feat=pos_dst_node_feat.squeeze(dim=-1)
            pos_node_src_id=pos_node_src_id.squeeze(dim=-1)
            pos_node_dst_id=pos_node_dst_id.squeeze(dim=-1)

            negative_score, neg_dst_node_feat, neg_node_src_id, neg_node_dst_id = model[1](negative_graph, nodes_representation, sampled_edge_type)
            negative_score=negative_score.squeeze(dim=-1)
            neg_dst_node_feat=neg_dst_node_feat.squeeze(dim=-1)
            neg_node_src_id=neg_node_src_id.squeeze(dim=-1)
            neg_node_dst_id=neg_node_dst_id.squeeze(dim=-1)

            train_y_predict = th.cat([positive_score, negative_score], dim=0)
            train_y_true = th.cat([th.ones_like(positive_score), th.zeros_like(negative_score)], dim=0)
            train_y_dst_node_representation = th.cat([pos_dst_node_feat, neg_dst_node_feat], dim=0)
            train_y_src_node_id = th.cat([pos_node_src_id, neg_node_src_id], dim=0)
            train_y_dst_node_id = th.cat([pos_node_dst_id, neg_node_dst_id], dim=0)

            loss = loss_func(train_y_predict, train_y_true)

            train_total_loss += loss.item()
            train_y_trues.append(train_y_true.detach().cpu())
            train_y_predicts.append(train_y_predict.detach().cpu())
            train_y_dst_node_representations.append(train_y_dst_node_representation.detach().cpu())
            train_y_src_node_ids.append(train_y_src_node_id.detach().cpu())
            train_y_dst_node_ids.append(train_y_dst_node_id.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)
        
        train_total_loss /= (batch + 1)
        train_y_trues = th.cat(train_y_trues, dim=0)
        train_y_predicts = th.cat(train_y_predicts, dim=0)
        train_y_dst_node_representations = th.cat(train_y_dst_node_representations, dim=0)
        train_y_src_node_ids = th.cat(train_y_src_node_ids, dim=0)
        train_y_dst_node_ids = th.cat(train_y_dst_node_ids, dim=0)
        train_RMSE, train_MAE, train_AUC, train_AP = evaluate_link_prediction(
            predict_scores=train_y_predicts, 
            true_scores=train_y_trues)
        train_HIT, train_DIV = evaluate_link_recommendation(
            predict_scores=train_y_predicts, 
            true_scores=train_y_trues,
            dst_node_representations=train_y_dst_node_representations,
            src_node_ids=train_y_src_node_ids,
            dst_node_ids=train_y_dst_node_ids)
        total_loss_vals['train'].append(train_total_loss)
        RMSE_vals['train'].append(train_RMSE)
        MAE_vals['train'].append(train_MAE)
        AUC_vals['train'].append(train_AUC)
        AP_vals['train'].append(train_AP)
        HIT_vals['train'].append(train_HIT)
        DIV_vals['train'].append(train_DIV)

        model.eval()
        val_total_loss, val_y_trues, val_y_predicts, val_y_dst_node_representations, val_y_src_node_ids, val_y_dst_node_ids = evaluate(
            model, 
            loader=val_loader, 
            loss_func=loss_func,
            sampled_edge_type=sampled_edge_type,
            device=args.device, 
            mode='validate')
        val_RMSE, val_MAE, val_AUC, val_AP = evaluate_link_prediction(
            predict_scores=val_y_predicts,
            true_scores=val_y_trues)
        val_HIT, val_DIV = evaluate_link_recommendation(
            predict_scores=val_y_predicts, 
            true_scores=val_y_trues,
            dst_node_representations=val_y_dst_node_representations,
            src_node_ids=val_y_src_node_ids,
            dst_node_ids=val_y_dst_node_ids)
        total_loss_vals['val'].append(val_total_loss)
        RMSE_vals['val'].append(val_RMSE)
        MAE_vals['val'].append(val_MAE)
        AUC_vals['val'].append(val_AUC)
        AP_vals['val'].append(val_AP)
        HIT_vals['val'].append(val_HIT)
        DIV_vals['val'].append(val_DIV)

        test_total_loss, test_y_trues, test_y_predicts, test_y_dst_node_representations, test_y_src_node_ids, test_y_dst_node_ids = evaluate(
            model, 
            loader=test_loader, 
            loss_func=loss_func,
            sampled_edge_type=sampled_edge_type,
            device=args.device, 
            mode='test')
        test_RMSE, test_MAE, test_AUC, test_AP = evaluate_link_prediction(
            predict_scores=test_y_predicts,
            true_scores=test_y_trues)
        test_HIT, test_DIV = evaluate_link_recommendation(
            predict_scores=test_y_predicts, 
            true_scores=test_y_trues,
            dst_node_representations=test_y_dst_node_representations,
            src_node_ids=test_y_src_node_ids,
            dst_node_ids=test_y_dst_node_ids)
        total_loss_vals['test'].append(test_total_loss)
        RMSE_vals['test'].append(test_RMSE)
        MAE_vals['test'].append(test_MAE)
        AUC_vals['test'].append(test_AUC)
        AP_vals['test'].append(test_AP)
        HIT_vals['test'].append(test_HIT)
        DIV_vals['test'].append(test_DIV)

        if best_validate_RMSE is None or val_RMSE < best_validate_RMSE:
            best_validate_RMSE = val_RMSE
            scores = {"RMSE": float(f"{test_RMSE:.4f}"), "MAE": float(f"{test_MAE:.4f}"),"AUC": float(f"{test_AUC:.4f}"), "AP": float(f"{test_AP:.4f}"), 'HIT': float(f"{test_HIT:.4f}"), 'DIV': float(f"{test_DIV:.4f}")}
            final_result = json.dumps(scores, indent=4)

        tqdm_loader.set_description(f'EPOCH #{epoch} RMSE: {test_RMSE:.4f}, MAE: {test_MAE:.4f}, AUC: {test_AUC:.4f}, AP: {test_AP:.4f}, HIT: {test_HIT:.4f}, DIV: {test_DIV:.4f} ')

        early_stop = early_stopping.step([('RMSE', val_RMSE, False), ('MAE', val_MAE, False)], model)

        if early_stop:
            break


    # save the model result
    save_result_dir= f"results/lfm1b/{date}/{sampled_edge_type}"
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)

    save_metrics_path = os.path.join(save_result_dir, f"metrics.json")
    with open(save_metrics_path, 'w') as file:
        file.write(final_result)
        file.close()

    save_args_path = os.path.join(save_result_dir, f"args.json")
    with open(save_args_path, 'w') as file:
        arg_dict=vars(args)
        arg_dict['model_parameters']=get_n_params(model)
        file.write(json.dumps(arg_dict, indent=4))
        file.close()

    make_loss_plots(sampled_edge_type, total_loss_vals, args.epochs, save_result_dir)
    make_RMSE_plots(sampled_edge_type, RMSE_vals, args.epochs, save_result_dir)
    make_MAE_plots(sampled_edge_type, MAE_vals, args.epochs, save_result_dir)
    make_AUC_plots(sampled_edge_type, AUC_vals, args.epochs, save_result_dir)
    make_AP_plots(sampled_edge_type, AP_vals, args.epochs, save_result_dir)
    make_HIT_plots(sampled_edge_type, HIT_vals, args.epochs, save_result_dir)
    make_DIV_plots(sampled_edge_type, DIV_vals, args.epochs, save_result_dir)

    print(f'save as {save_result_dir}')
    print(f"predicted relation: {sampled_edge_type}")
    print(f'result: {final_result}')



def train_models(hg, args):
    th.cuda.empty_cache()
    print('\n')
    print('Training Process Started')
    print('running with args','\n')
    print(args)
    
    link_score_predictor = LinkScorePredictor(args.hidden_dim * args.num_heads)
    warnings.filterwarnings('ignore')
    set_random_seed(args.seed)
    
    reverse_etypes = dict()
    for stype, etype, dtype in hg.canonical_etypes: 
        for srctype, reltype, dsttype in hg.canonical_etypes:
            if srctype == dtype and dsttype == stype and reltype != etype:
                reverse_etypes[etype] = reltype
                break
    r_hgnn = R_HGNN(graph=hg,
                    input_dim_dict={ntype: hg.nodes[ntype].data['feat'].shape[1] for ntype in hg.ntypes},
                    hidden_dim=args.hidden_dim, 
                    relation_input_dim=args.rel_input_dim,
                    relation_hidden_dim=args.rel_hidden_dim,
                    num_layers=args.num_layers, 
                    n_heads=args.num_heads, 
                    dropout=args.dropout,
                    residual=args.residual, 
                    norm=args.norm)
    date=datetime.now().strftime("%d_%m_%Y_%H:%M:%S").replace(' ',"_")
    for sampled_edge_type in [etype for _,etype,_ in hg.canonical_etypes]:

        if sampled_edge_type in ['listened_to_track','listened_to_album','listened_to_artist']:
            print('\n','Training',sampled_edge_type,'Model')
            print(hg)
            th.cuda.empty_cache()
            save_model_folder = f"results/lfm1b/{date}/{sampled_edge_type}"
            train_edge_idx, valid_edge_idx, test_edge_idx = get_predict_edge_index(
                hg,
                sample_edge_rate=args.sample_edge_rate,
                sampled_edge_type=sampled_edge_type,
                seed=args.seed)

            train_loader, val_loader, test_loader = get_edge_data_loader(
                args.node_min_neighbors,
                args.num_layers,
                hg,
                args.batch_size,
                sampled_edge_type,
                args.num_neg_samples,
                train_edge_idx=train_edge_idx,
                valid_edge_idx=valid_edge_idx,
                test_edge_idx=test_edge_idx,
                reverse_etypes=reverse_etypes,
                shuffle = args.shuffle, 
                drop_last = args.drop_last,
                num_workers = args.num_workers
                )
            del train_edge_idx
            del valid_edge_idx
            del test_edge_idx

            model = nn.Sequential(r_hgnn, link_score_predictor)
            model = convert_to_gpu(model, device=args.device)
            print('len(train_loader)',len(train_loader),'len(val_loader)',len(val_loader),'len(test_loader)',len(test_loader))
            print(f'Model #Params: {get_n_params(model)}')
            optimizer, scheduler = get_optimizer_and_lr_scheduler(
                model, 
                args.opt, 
                args.learning_rate, 
                args.weight_decay,
                steps_per_epoch=len(train_loader), 
                epochs=args.epochs)
            print(f'Training {sampled_edge_type} link prediction model')
            train_model(model, optimizer,scheduler, train_loader, val_loader, test_loader, save_model_folder, sampled_edge_type, date, args)
    del model
    del optimizer
    del scheduler
    del train_loader
    del val_loader
    del test_loader
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    parser.add_argument('--sample_edge_rate', default=0.1, type=float, help='train: validate: test ratio')
    parser.add_argument('--num_layers', default=2, type=int, help='number of convolutional layers for a model')
    parser.add_argument('--batch_size', default=512, type=int, help='the number of edges to train in each batch')
    parser.add_argument('--num_neg_samples', default=5, type=int, help='the number of negative edges to sample when training')
    parser.add_argument('--node_min_neighbors', default=5, type=int, help='the number of nodes to sample per target node')
    parser.add_argument('--shuffle',  default=True, type=str2bool, nargs='?', const=True, help='string bool wether to shuffle indicies before splitting')
    parser.add_argument('--drop_last',  default=False, type=str2bool, nargs='?', const=True, help='string bool wether to drop the last sample in data loading')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers for a specified data loader')
    parser.add_argument('--hidden_dim', default=32, type=int, help='dimension of the hidden layer input')
    parser.add_argument('--rel_input_dim', default=12, type=int, help='input dimension of the edges')
    parser.add_argument('--rel_hidden_dim', default=32, type=int, help='hidden dimension of the edges')
    parser.add_argument('--num_heads', default=8, type=int, help='the number of attention heads used')
    parser.add_argument('--dropout', default=0.5, type=float, help='the dropout rate for the models')
    parser.add_argument('--residual', default=True, type=str2bool, nargs='?', const=True, help='string for using the residual values in computation')
    parser.add_argument('--norm', default=True, type=str2bool, nargs='?', const=True, help=' string for using normalization of values in computation')
    parser.add_argument('--opt', default='adam', type=str, help='the name of the optimizer to be used')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='the decay of the weights used for training')
    parser.add_argument('--epochs', default=100, type=int, help='the number of epochs to train the model with')
    parser.add_argument('--patience', default=25, type=int, help='the number of epochs to allow before early stopping')
    parser.add_argument('--split_by_users', default=False, type=str2bool, nargs='?', const=True, help='boolean inidicator if you want to split train/val/test by users and not just targetedges')
    parser.add_argument('--n_users', default=None, type=str, help="number of LE rows rto collect for a subset of the full dataset")
    parser.add_argument('--popular_artists', default=False, type=str2bool, nargs='?', const=True, help="number of LE rows rto collect for a subset of the full dataset")
    parser.add_argument('--device', default='cpu', type=str, help='GPU or CPU device specification')
    parser.add_argument('--overwrite_preprocessed', default=False, type=str2bool, nargs='?', const=True, help='indication to overwrite preprocessed ')
    parser.add_argument('--overwrite_processed', default=False, type=str2bool, nargs='?', const=True, help='indication to overwrite processed')
    parser.add_argument('--artists', default=True, type=str2bool, nargs='?', const=True, help='indication to use the artist and genre nodes in the graph')
    parser.add_argument('--albums', default=True, type=str2bool, nargs='?', const=True, help='indication to use the albums and genre nodes in the graph')
    parser.add_argument('--tracks', default=True, type=str2bool, nargs='?', const=True, help='indication to use the tracks and genre nodes in the graph')
    parser.add_argument('--playcount_weight', default=False, type=str2bool, nargs='?', const=True, help='indication give every edge a "playcount weight" feature, or every edge with "timestamp" features between a user and their unique listen events')
    parser.add_argument('--norm_playcount_weight', default=True, type=str2bool, nargs='?', const=True, help='indication give every edge a "normalized playcount weight" feature, or "total playcount weight"')
    parser.add_argument('--metapath2vec', default=True, type=str2bool, nargs='?', const=True, help='indication to use metapath2vec to encode node embeddings (recommended, otherwise manual adjustment may be required)')
    parser.add_argument('--emb_dim', default=32, type=int,  help='node embedding vector size')
    parser.add_argument('--walk_length', default=64, type=int,  help='length of metapath2vec walks')
    parser.add_argument('--context_size', default=7, type=int,  help='context_size of metapath2vec')
    parser.add_argument('--walks_per_node', default=3, type=int,  help='context_size of metapath2vec')
    parser.add_argument('--metapath2vec_epochs_batch_size', default=512, type=int,  help='batch_size of metapath2vec')
    parser.add_argument('--learning_rate', default=0.001, type=float,  help='learning_rate of metapath2vec')
    parser.add_argument('--metapath2vec_epochs', default=5, type=int,  help='epochs of metapath2vec')
    parser.add_argument('--logs', default=100, type=int,  help='logs of metapath2vec')
 
    args = parser.parse_args()

    dataset = LFM1b(
        n_users=args.n_users, 
        popular_artists=args.popular_artists,
        device=args.device, 
        overwrite_preprocessed=args.overwrite_preprocessed,
        overwrite_processed=args.overwrite_processed,
        artists=args.artists,
        albums=args.albums,
        tracks=args.tracks,
        playcount_weight=args.playcount_weight,
        norm_playcount_weight=args.norm_playcount_weight,
        metapath2vec=args.metapath2vec,
        emb_dim=args.emb_dim, 
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        num_negative_samples=args.num_neg_samples,
        batch_size=args.metapath2vec_epochs_batch_size,
        learning_rate=args.learning_rate,
        epochs=args.metapath2vec_epochs,
        logs=args.logs
        )
    print('Loading graph')
    glist,_= dataset.load()
    hg=glist[0] 
    print(hg)

    train_models(hg, args)
    