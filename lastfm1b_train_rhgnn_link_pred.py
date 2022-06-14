import os 
import torch as th 
import torch.nn as nn
from tqdm import tqdm
import copy
import json
import shutil
import warnings
from utils.utils import set_random_seed, get_edge_data_loader, get_predict_edge_index, convert_to_gpu, get_n_params, get_optimizer_and_lr_scheduler, evaluate_link_prediction
from dgl.data.utils import load_graphs
from utils.LinkScorePredictor import LinkScorePredictor
from utils.EarlyStopping import EarlyStopping
from model.R_HGNN import R_HGNN


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
        loader_tqdm = tqdm(loader, ncols=120)
        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(loader_tqdm):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(positive_graph, negative_graph, device=device)
            # target node relation representation in the heterogeneous graph
            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                              blocks[0].canonical_etypes}

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

            loader_tqdm.set_description(f'{mode} for the {batch}-th batch, {mode} loss: {loss.item()}')

        total_loss /= (batch + 1)
        y_trues = th.cat(y_trues, dim=0)
        y_predicts = th.cat(y_predicts, dim=0)

    return total_loss, y_trues, y_predicts


def train_model(model, optimizer,scheduler, train_loader, epochs, save_folder, model_name, sampled_edge_type, DEVICE):
    
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)
    patience = 50
    early_stopping = EarlyStopping(
    patience=patience, 
    save_model_folder=save_folder,
    save_model_name=model_name)

    loss_func = nn.BCELoss()
    train_steps = 0
    best_validate_RMSE, final_result = None, None
    loss_values=[]

    for epoch in range(epochs):
        model.train()
        train_y_trues = []
        train_y_predicts = []
        train_total_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, ncols=120)

        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_loader_tqdm):
            blocks = [convert_to_gpu(b, device=DEVICE) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(
                positive_graph, 
                negative_graph, device=DEVICE)

            # target node relation representation in the heterogeneous graph
            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in blocks[0].canonical_etypes}
            # for k,v in input_features.items():
            #     print(k,v.shape)
            nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features))
            # for k,v in nodes_representation.items():
            #     print(k,v.shape)
            
            positive_score = model[1](
                positive_graph, 
                nodes_representation, 
                sampled_edge_type).squeeze(dim=-1)
            negative_score = model[1](
                negative_graph, 
                nodes_representation, 
                sampled_edge_type).squeeze(dim=-1)


            train_y_predict = th.cat([positive_score, negative_score], dim=0)
            train_y_true = th.cat(
                [th.ones_like(positive_score), 
                th.zeros_like(negative_score)], dim=0)
            loss = loss_func(train_y_predict, train_y_true)

            train_total_loss += loss.item()
            loss_values.append(loss.item())
            train_y_trues.append(train_y_true.detach().cpu())
            train_y_predicts.append(train_y_predict.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loader_tqdm.set_description(f'training for the {batch}-th batch, train loss: {loss.item()}')

            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)

        train_total_loss /= (batch + 1)
        train_y_trues = th.cat(train_y_trues, dim=0)
        train_y_predicts = th.cat(train_y_predicts, dim=0)

        train_RMSE, train_MAE = evaluate_link_prediction(
            predict_scores=train_y_predicts, 
            true_scores=train_y_trues)

        model.eval()

        val_total_loss, val_y_trues, val_y_predicts = evaluate(
            model, 
            loader=val_loader, 
            loss_func=loss_func,
            sampled_edge_type=sampled_edge_type,
            device=DEVICE, 
            mode='validate')

        val_RMSE, val_MAE = evaluate_link_prediction(
            predict_scores=val_y_predicts,
            true_scores=val_y_trues)

        test_total_loss, test_y_trues, test_y_predicts = evaluate(
            model, 
            loader=test_loader, 
            loss_func=loss_func,
            sampled_edge_type=sampled_edge_type,
            device=DEVICE, 
            mode='test')

        test_RMSE, test_MAE = evaluate_link_prediction(
            predict_scores=test_y_predicts,
            true_scores=test_y_trues)

        if best_validate_RMSE is None or val_RMSE < best_validate_RMSE:
            best_validate_RMSE = val_RMSE
            scores = {"RMSE": float(f"{test_RMSE:.4f}"), "MAE": float(f"{test_MAE:.4f}")}
            final_result = json.dumps(scores, indent=4)

        print(
            f'Epoch: {epoch}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}, RMSE {train_RMSE:.4f}, MAE {train_MAE:.4f}, \n'
            f'validate loss: {val_total_loss:.4f}, RMSE {val_RMSE:.4f}, MAE {val_MAE:.4f}, \n'
            f'test loss: {test_total_loss:.4f}, RMSE {test_RMSE:.4f}, MAE {test_MAE:.4f}')

        early_stop = early_stopping.step([('RMSE', val_RMSE, False), ('MAE', val_MAE, False)], model)

        if early_stop:
            break


    # save the model result

    SAVE_RESULT_FOLDER= f"../results/lfm1b_{sampled_edge_type}"
    if not os.path.exists(SAVE_RESULT_FOLDER):
        os.makedirs(SAVE_RESULT_FOLDER, exist_ok=True)
    save_result_path = os.path.join(SAVE_RESULT_FOLDER, f"{model_name}.json")

    with open(save_result_path, 'w') as file:
        file.write(final_result)
        file.close()

    print(f'save as {save_result_path}')
    print(f"predicted relation: {sampled_edge_type}")
    print(f'result: {final_result}')


## ARGS
SEED=0
SAMPLE_EDGE_RATE=0.01
NODE_NEIGHTBORS_MIN_NUM = 10
N_LAYERS = 2
BATCH_SIZE = 1024
NEGATIVE_SAMPLE_EDGE_NUM= 5
SHUFFLE = True
DROP_LAST = False
NUM_WORKERS = 4
HIDDEN_DIM = 32
RELATIONAL_INPUT_DIM = 20
RELATIONAL_HIDDEN_DIM = 8
N_HEADS = 8
DROPOUT = 0.3
RESIDUAL = True
NORM = True
HID_DIM = 32
N_HEADS = 8
LINK_SCORE_PREDICTOR = LinkScorePredictor(HID_DIM * N_HEADS)
OPTIMIZER_NAME = 'adam'
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
EPOCHS = 200
DEVICE='cuda'

warnings.filterwarnings('ignore')
set_random_seed(SEED)

# using DGl's load_graphs function to load pre-computed and processed files
glist,_=load_graphs('lastfm1b_subset.bin') # <- this file represents a subset of the full dataset
hg=glist[0] # hg=='heterogeneous graph' ;) from the list of graphs in the processed file (hint: theres only one) pick our heterogenous subset graph

# creating a dictionary of every edge and it's reverse edge
reverse_etypes = dict()
for stype, etype, dtype in hg.canonical_etypes: # for every edge type structured as (phi(u), psi(e), phi(v))
    for srctype, reltype, dsttype in hg.canonical_etypes:
        if srctype == dtype and dsttype == stype and reltype != etype:
            reverse_etypes[etype] = reltype
            break

r_hgnn = R_HGNN(graph=hg,
                input_dim_dict={ntype: hg.nodes[ntype].data['feat'].shape[1] for ntype in hg.ntypes},
                hidden_dim=HIDDEN_DIM, 
                relation_input_dim=RELATIONAL_INPUT_DIM,
                relation_hidden_dim=RELATIONAL_HIDDEN_DIM,
                num_layers=N_LAYERS, 
                n_heads=N_HEADS, 
                dropout=DROPOUT,
                residual=RESIDUAL, 
                norm=NORM)

for SAMPLED_EDGE_TYPE in ['listened_to_artist','listened_to_album','listened_to_track']:
    MODEL_NAME='R_HGNN'+'_'+SAMPLED_EDGE_TYPE
    SAVE_MODEL_FOLDER = f"../save_model/'lfm1b'/{MODEL_NAME}"
    train_edge_idx, valid_edge_idx, test_edge_idx = get_predict_edge_index(
        hg,
        sample_edge_rate=SAMPLE_EDGE_RATE,
        sampled_edge_type=SAMPLED_EDGE_TYPE,
        seed=SEED)

    train_loader, val_loader, test_loader = get_edge_data_loader(
        NODE_NEIGHTBORS_MIN_NUM,
        N_LAYERS,
        hg,
        BATCH_SIZE,
        SAMPLED_EDGE_TYPE,
        NEGATIVE_SAMPLE_EDGE_NUM,
        train_edge_idx=train_edge_idx,
        valid_edge_idx=valid_edge_idx,
        test_edge_idx=test_edge_idx,
        reverse_etypes=reverse_etypes,
        shuffle = SHUFFLE, 
        drop_last = DROP_LAST,
        num_workers = NUM_WORKERS
        )

    model = nn.Sequential(r_hgnn, LINK_SCORE_PREDICTOR)
    model = convert_to_gpu(model, device=DEVICE)

    print(f'{SAMPLED_EDGE_TYPE} Model #Params: {get_n_params(model)}.')
    optimizer, scheduler = get_optimizer_and_lr_scheduler(
        model, 
        OPTIMIZER_NAME, 
        LEARNING_RATE, 
        WEIGHT_DECAY,
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS)


    train_model(model, optimizer,scheduler, train_loader, EPOCHS, SAVE_MODEL_FOLDER, MODEL_NAME, SAMPLED_EDGE_TYPE, DEVICE)


