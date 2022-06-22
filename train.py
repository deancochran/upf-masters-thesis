import argparse
import copy
import json
import os
import shutil
from signal import raise_signal
import warnings
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import torch as th 
import torch.nn as nn
from tqdm import tqdm
from LFM1b import LFM1b
from utils.LinkScorePredictor import LinkScorePredictor
from utils.EarlyStopping import EarlyStopping
from model.R_HGNN import R_HGNN
from utils.utils import evaluate_link_prediction, get_predict_edge_index, set_random_seed, get_edge_data_loader, convert_to_gpu, get_optimizer_and_lr_scheduler, get_n_params
th.cuda.empty_cache()


def make_loss_plots(sample_edge_type, total_loss_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(total_loss_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), total_loss_vals['train']+zero_pad, 'r', label='Training loss')
    plt.plot(range(epochs), total_loss_vals['val']+zero_pad, 'g', label='Val loss')
    plt.plot(range(epochs), total_loss_vals['test']+zero_pad, 'b', label='Test loss')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim((0, 2))
    plt.yticks(np.arange(0, 2, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_loss_plot.png")

def make_RMSE_plots(sample_edge_type, RMSE_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(RMSE_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), RMSE_vals['train']+zero_pad, 'r', label='Training RMSE')
    plt.plot(range(epochs), RMSE_vals['val']+zero_pad, 'g', label='Val RMSE')
    plt.plot(range(epochs), RMSE_vals['test']+zero_pad, 'b', label='Test RMSE')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.ylim((0, 2))
    plt.yticks(np.arange(0, 2, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_rmse_plot.png")

def make_MAE_plots(sample_edge_type, MAE_vals, epochs, save_result_dir):
    zero_pad=[0 for epoch in range(epochs-len(MAE_vals['train']))]
    plt.figure()
    plt.plot(range(epochs), MAE_vals['train']+zero_pad, 'r', label='Training MAE')
    plt.plot(range(epochs), MAE_vals['val']+zero_pad, 'g', label='Val MAE')
    plt.plot(range(epochs), MAE_vals['test']+zero_pad, 'b', label='Test MAE')
    plt.title(f'{sample_edge_type} link-pred Train, Val, Test MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.ylim((0, 2))
    plt.yticks(np.arange(0, 2, step=0.2))
    plt.legend()
    plt.savefig(save_result_dir+f"/{sample_edge_type}_mae_plot.png")


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
        total_loss = 0.0
        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(loader):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(positive_graph, negative_graph, device=device)
            # target node relation representation in the heterogeneous graph
            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                              blocks[0].canonical_etypes}

            # lb = LabelBinarizer()
            # labels = [etype for _,etype,_ in blocks[0].canonical_etypes]
            # lb.fit(labels)
            # relational_embedding={etype: th.tensor([float(val) for val in vector], device=args.device) for etype,vector in zip(labels, lb.transform(labels))}
            # nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features), copy.deepcopy(relational_embedding))

            nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features))

            positive_score = model[1](
                positive_graph, 
                nodes_representation, 
                sampled_edge_type).squeeze(dim=-1)
            negative_score = model[1](
                negative_graph, 
                nodes_representation, 
                sampled_edge_type).squeeze(dim=-1)

            y_predict = th.cat([positive_score, negative_score], dim=0)
            y_true = th.cat(
                [th.ones_like(positive_score), 
                th.zeros_like(negative_score)], dim=0)

            loss = loss_func(y_predict, y_true)
            total_loss += loss.item()
            y_trues.append(y_true.detach().cpu())
            y_predicts.append(y_predict.detach().cpu())

        total_loss /= (batch + 1)
        y_trues = th.cat(y_trues, dim=0)
        y_predicts = th.cat(y_predicts, dim=0)

    return total_loss, y_trues, y_predicts

def train_model(model, optimizer,scheduler, train_loader, val_loader, test_loader, save_folder, sample_edge_type, date, args):
    
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_folder, save_model_name=sample_edge_type)
    tqdm_loader = tqdm(range(args.epochs), total=args.epochs)
    loss_func = nn.BCELoss()
    train_steps = 0
    best_validate_RMSE, final_result = None, None

    total_loss_vals={'train':[],'val':[],'test':[]}
    RMSE_vals={'train':[],'val':[],'test':[]}
    MAE_vals={'train':[],'val':[],'test':[]}
    for epoch in tqdm_loader:
        model.train()
        train_y_trues = []
        train_y_predicts = []
        train_total_loss = 0.0
        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_loader):
            blocks = [convert_to_gpu(b, device=args.device) for b in blocks]
            blocks = [convert_to_gpu(b, device=args.device) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(positive_graph, negative_graph, device=args.device)
            
            input_features = {}
            # for (stype, etype, dtype) in blocks[0].canonical_etypes:
            #     # blocks[0].srcnodes[dtype].data['feat'] =
            #     # input_features[(stype, etype, dtype)]
            #     pass
            
            # lb = LabelBinarizer()
            # labels = [etype for _,etype,_ in blocks[0].canonical_etypes]
            # lb.fit(labels)
            # relational_embedding={etype: th.tensor([float(val) for val in vector], device=args.device) for etype,vector in zip(labels, lb.transform(labels))}
            # nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features), copy.deepcopy(relational_embedding))

            nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features))

            positive_score = model[1](positive_graph, nodes_representation, sample_edge_type).squeeze(dim=-1)
            negative_score = model[1](negative_graph, nodes_representation, sample_edge_type).squeeze(dim=-1)

            train_y_predict = th.cat([positive_score, negative_score], dim=0)
            train_y_true = th.cat([th.ones_like(positive_score), th.zeros_like(negative_score)], dim=0)
            loss = loss_func(train_y_predict, train_y_true)
            train_total_loss += loss.item()
            train_y_trues.append(train_y_true.detach().cpu())
            train_y_predicts.append(train_y_predict.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)
        
        train_total_loss /= (batch + 1)
        train_y_trues = th.cat(train_y_trues, dim=0)
        train_y_predicts = th.cat(train_y_predicts, dim=0)
        train_RMSE, train_MAE = evaluate_link_prediction(
            predict_scores=train_y_predicts, 
            true_scores=train_y_trues)
        total_loss_vals['train'].append(train_total_loss)
        RMSE_vals['train'].append(train_RMSE)
        MAE_vals['train'].append(train_MAE)

        model.eval()

        val_total_loss, val_y_trues, val_y_predicts = evaluate(
            model, 
            loader=val_loader, 
            loss_func=loss_func,
            sampled_edge_type=sample_edge_type,
            device=args.device, 
            mode='validate')
        val_RMSE, val_MAE = evaluate_link_prediction(
            predict_scores=val_y_predicts,
            true_scores=val_y_trues)
        total_loss_vals['val'].append(val_total_loss)
        RMSE_vals['val'].append(val_RMSE)
        MAE_vals['val'].append(val_MAE)

        test_total_loss, test_y_trues, test_y_predicts = evaluate(
            model, 
            loader=test_loader, 
            loss_func=loss_func,
            sampled_edge_type=sample_edge_type,
            device=args.device, 
            mode='test')
        test_RMSE, test_MAE = evaluate_link_prediction(
            predict_scores=test_y_predicts,
            true_scores=test_y_trues)
        total_loss_vals['test'].append(test_total_loss)
        RMSE_vals['test'].append(test_RMSE)
        MAE_vals['test'].append(test_MAE)

        if best_validate_RMSE is None or val_RMSE < best_validate_RMSE:
            best_validate_RMSE = val_RMSE
            scores = {"RMSE": float(f"{test_RMSE:.4f}"), "MAE": float(f"{test_MAE:.4f}")}
            final_result = json.dumps(scores, indent=4)

        tqdm_loader.set_description(f'EPOCH #{epoch} train loss: {train_total_loss:.4f}, validate loss: {val_total_loss:.4f}, train RMSE: {train_RMSE:.4f}, validate RMSE: {val_RMSE:.4f} ')
        # print(
        #     f'Epoch: {epoch}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}, RMSE {train_RMSE:.4f}, MAE {train_MAE:.4f}, \n'
        #     f'validate loss: {val_total_loss:.4f}, RMSE {val_RMSE:.4f}, MAE {val_MAE:.4f}, \n'
        #     f'test loss: {test_total_loss:.4f}, RMSE {test_RMSE:.4f}, MAE {test_MAE:.4f}')

        early_stop = early_stopping.step([('RMSE', val_RMSE, False), ('MAE', val_MAE, False)], model)

        if early_stop:
            break


    # save the model result
    save_result_dir= f"results/lfm1b/{date}/{sample_edge_type}"
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)

    save_metrics_path = os.path.join(save_result_dir, f"metrics.json")
    with open(save_metrics_path, 'w') as file:
        file.write(final_result)
        file.close()

    save_args_path = os.path.join(save_result_dir, f"args.json")
    with open(save_args_path, 'w') as file:
        file.write(json.dumps(vars(args), indent=4))
        file.close()

    make_loss_plots(sample_edge_type, total_loss_vals, args.epochs, save_result_dir)
    make_RMSE_plots(sample_edge_type, RMSE_vals, args.epochs, save_result_dir)
    make_MAE_plots(sample_edge_type, MAE_vals, args.epochs, save_result_dir)

    print(f'save as {save_result_dir}')
    print(f"predicted relation: {sample_edge_type}")
    print(f'result: {final_result}')



def train_models(args):
    print('Training Process Started')
    print('\n')

    link_score_predictor = LinkScorePredictor(args.hidden_dim * args.num_heads)
    warnings.filterwarnings('ignore')
    set_random_seed(args.seed)

    # using DGl's load_graphs function to load pre-computed and processed files
    
    dataset=LFM1b(n_users=args.n_users, overwrite_raw=args.overwrite_raw, overwrite_preprocessed=args.overwrite_preprocessed, overwrite_processed=args.overwrite_processed)
    print('Loading graph')
    glist,_= dataset.load() # <- this file represents a subset of the full dataset
    hg=glist[0] # hg=='heterogeneous graph' ;) from the list of graphs in the processed file (hint: theres only one) pick our heterogenous subset graph
    # print('Dataset:')
    # print('\n')
    print(hg)
    print('\n')

    # creating a dictionary of every edge and it's reverse edge
    reverse_etypes = dict()
    for stype, etype, dtype in hg.canonical_etypes: # for every edge type structured as (phi(u), psi(e), phi(v))
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
    for sample_edge_type in ['listened_to_artist','listened_to_album','listened_to_track']:
        save_model_folder = f"results/lfm1b/{date}/{sample_edge_type}"
        train_edge_idx, valid_edge_idx, test_edge_idx = get_predict_edge_index(
            hg,
            sample_edge_rate=args.sample_edge_rate,
            sampled_edge_type=sample_edge_type,
            seed=args.seed)

        train_loader, val_loader, test_loader = get_edge_data_loader(
            args.node_min_neighbors,
            args.num_layers,
            hg,
            args.batch_size,
            sample_edge_type,
            args.num_neg_samples,
            train_edge_idx=train_edge_idx,
            valid_edge_idx=valid_edge_idx,
            test_edge_idx=test_edge_idx,
            reverse_etypes=reverse_etypes,
            shuffle = args.shuffle, 
            drop_last = args.drop_last,
            num_workers = args.num_workers
            )

        model = nn.Sequential(r_hgnn, link_score_predictor)
        model = convert_to_gpu(model, device=args.device)
        print(f'Model #Params: {get_n_params(model)}.')
        print('len(train_loader)',len(train_loader))
        print('len(val_loader)',len(val_loader))
        print('len(test_loader)',len(test_loader))

        # print(f'{sample_edge_type} Model #Params: {get_n_params(model)}.')
        optimizer, scheduler = get_optimizer_and_lr_scheduler(
            model, 
            args.opt, 
            args.learing_rate, 
            args.weight_decay,
            steps_per_epoch=len(train_loader), 
            epochs=args.epochs)
        print(f'Training {sample_edge_type} link prediction model')
        train_model(model, optimizer,scheduler, train_loader, val_loader, test_loader, save_model_folder, sample_edge_type, date, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', default=None, type=int, help='subset parameter')
    parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
    parser.add_argument('--sample_edge_rate', default=0.05, type=float, help='train: validate: test ratio')
    parser.add_argument('--num_layers', default=2, type=int, help='number of convolutional layers for a model')
    parser.add_argument('--batch_size', default=512, type=int, help='the number of edges to train in each batch')
    parser.add_argument('--num_neg_samples', default=5, type=int, help='the number of negative edges to sample when training')
    parser.add_argument('--node_min_neighbors', default=10, type=int, help='the number of nodes to sample per target node')
    parser.add_argument('--shuffle',  default=True, type=bool, help='wether to shuffle indicies before splitting')
    parser.add_argument('--drop_last',  default=False, type=bool, help='wether to drop the last sample in data loading')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for a specified data loader')
    parser.add_argument('--hidden_dim', default=32, type=int, help='dimension of the hidden layer input')
    parser.add_argument('--rel_input_dim', default=12, type=int, help='input dimension of the edges')
    parser.add_argument('--rel_hidden_dim', default=32, type=int, help='hidden dimension of the edges')
    parser.add_argument('--num_heads', default=12, type=int, help='the number of attention heads used')
    parser.add_argument('--dropout', default=0.5, type=float, help='the dropout rate for the models')
    parser.add_argument('--residual', default=True, type=bool, help='using the residual values in computation')
    parser.add_argument('--norm', default=True, type=bool, help='using normalization of values in computation')
    parser.add_argument('--opt', default='adam', type=str, help='the name of the optimizer to be used')
    parser.add_argument('--learing_rate', default=0.001, type=float, help='the learning rate used for training')
    parser.add_argument('--weight_decay', default=0.00, type=float, help='the decay of the weights used for training')
    parser.add_argument('--epochs', default=200, type=int, help='the number of epochs to train the model with')
    parser.add_argument('--device', default='cuda', type=str, help='the gpu device used for computation')
    parser.add_argument('--patience', default=50, type=int, help='the number of epochs to allow before early stopping')
    parser.add_argument('--overwrite_raw', default=False, type=bool, help='overwrites the original data collection by unzipping the zip file')
    parser.add_argument('--overwrite_preprocessed', default=False, type=bool, help='overwrites the preprocessed data by running dataset loader')
    parser.add_argument('--overwrite_processed', default=False, type=bool, help='overwrites processed graph file, by compiling graph')
    args = parser.parse_args()
 
    train_models(args)
    