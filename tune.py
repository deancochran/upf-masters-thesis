import itertools
from msilib.schema import Error
import os
import json
import pandas as pd
import torch as th
from datetime import datetime
from train import train_models
from DGL_LFM1b.DGL_LFM1b import LFM1b
th.cuda.empty_cache()

# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

args_range_dict={
    'seed':[0],
    'num_layers':[2,4],
    'sample_edge_rate':[0.05,0.01],
    'batch_size':[256,1024],
    'num_neg_samples': [5],
    'node_min_neighbors':[20],
    'shuffle':[True],
    'drop_last':[False],
    'num_workers':[4],
    'hidden_dim':[32],
    'rel_input_dim':[16],
    'rel_hidden_dim':[32],
    'num_heads':[8],
    'dropout':[.05,.075],
    'residual':[True, False],
    'norm':[True],
    'opt':['adam'],
    'learing_rate':[.001],
    'weight_decay':[0.0],
    'epochs':[200],
    'device':['cpu'],
    'patience':[25],
    'overwrite':[False],
}

# parser.add_argument('--seed', default=0, type=int, help='seed for reproducibility')
# parser.add_argument('--sample_edge_rate', default=0.01, type=float, help='train: validate: test ratio')
# parser.add_argument('--num_layers', default=2, type=int, help='number of convolutional layers for a model')
# parser.add_argument('--batch_size', default=512, type=int, help='the number of edges to train in each batch')
# parser.add_argument('--num_neg_samples', default=5, type=int, help='the number of negative edges to sample when training')
# parser.add_argument('--node_min_neighbors', default=10, type=int, help='the number of nodes to sample per target node')
# parser.add_argument('--shuffle',  default=True, type=str2bool, nargs='?', const=True, help='string bool wether to shuffle indicies before splitting')
# parser.add_argument('--drop_last',  default=False, type=str2bool, nargs='?', const=True, help='string bool wether to drop the last sample in data loading')
# parser.add_argument('--num_workers', default=4, type=int, help='number of workers for a specified data loader')
# parser.add_argument('--hidden_dim', default=32, type=int, help='dimension of the hidden layer input')
# parser.add_argument('--rel_input_dim', default=16, type=int, help='input dimension of the edges')
# parser.add_argument('--rel_hidden_dim', default=32, type=int, help='hidden dimension of the edges')
# parser.add_argument('--num_heads', default=8, type=int, help='the number of attention heads used')
# parser.add_argument('--dropout', default=0.8, type=float, help='the dropout rate for the models')
# parser.add_argument('--residual', default=True, type=str2bool, nargs='?', const=True, help='string for using the residual values in computation')
# parser.add_argument('--norm', default=True, type=str2bool, nargs='?', const=True, help=' string for using normalization of values in computation')
# parser.add_argument('--opt', default='adam', type=str, help='the name of the optimizer to be used')
# parser.add_argument('--learing_rate', default=0.001, type=float, help='the learning rate used for training')
# parser.add_argument('--weight_decay', default=0.00, type=float, help='the decay of the weights used for training')
# parser.add_argument('--epochs', default=200, type=int, help='the number of epochs to train the model with')
# parser.add_argument('--device', default='cpu', type=str, help='the gpu device used for computation')
# parser.add_argument('--patience', default=50, type=int, help='the number of epochs to allow before early stopping')
# parser.add_argument('--name', default='DGL_LFM1b', type=str, help='name of directory in data folder')
# parser.add_argument('--n_users', default=None, type=str, help="number of LE rows rto collect for a subset of the full dataset")
# parser.add_argument('--overwrite_preprocessed', default=False, type=str2bool, nargs='?', const=True, help='string indication wheter to overwrite preprocessed ')
# parser.add_argument('--overwrite_processed', default=False, type=str2bool, nargs='?', const=True, help='string indication wheter to overwrite processed')
# parser.add_argument('--artists', default=True, type=str2bool, nargs='?', const=True, help='string indication wheter to use the artist and genre nodes in the graph')
# parser.add_argument('--albums', default=True, type=str2bool, nargs='?', const=True, help='string indication wheter to use the albums and genre nodes in the graph')
# parser.add_argument('--tracks', default=True, type=str2bool, nargs='?', const=True, help='string indication wheter to use the tracks and genre nodes in the graph')
# parser.add_argument('--playcount_weight', default=False, type=str2bool, nargs='?', const=True, help='Specifiy whether or not the weighted edge playcount connection between users and their listened artists/tracks/albums is applied')
 





print('\n')
print('Tuning Hyperparameters')

keys, values = zip(*args_range_dict.items())
permutations_args = [dict(zip(keys, v)) for v in itertools.product(*values)]
for n_users in args_range_dict['n_users']:
    for i, arg_perm in enumerate(permutations_args):
        args=Dict2Class(arg_perm)
        if args.n_users == n_users:
            if (i==0):
                args.overwrite=True
                dataset=LFM1b(n_users=args.n_users, overwrite=args.overwrite)
                print('Loading graph')
                glist,_= dataset.load()
                hg=glist[0] 
            try:
                train_models(hg, args)
            except Error as e:
                print(e)
            
    del hg

    




















results_dir='results/lfm1b'
results_data={
    'model':[],
    'time':[],
    'path_to_loss_img':[],
    'path_to_mae_img':[],
    'path_to_rmse_img':[],
    'path_to_pkl':[],
    'RMSE':[],
    'MAE':[],
    'nrows':[],
    'num_layers':[],
    'sample_edge_rate':[],
    'batch_size':[],
    'num_neg_samples':[],
    'node_min_neighbors':[],
    'shuffle':[],
    'drop_last':[],
    'num_workers':[],
    'hidden_dim':[],
    'rel_input_dim':[],
    'rel_hidden_dim':[],
    'num_heads':[],
    'dropout':[],
    'residual':[],
    'norm':[],
    'opt':[],
    'learing_rate':[],
    'weight_decay':[],
    'epochs':[],
    'device':[],
    'patience':[],
    'overwrite_raw':[],
    'overwrite_preprocessed':[],
    'overwrite_processed':[]
}
for result_folder in os.listdir(results_dir):
    print(result_folder)
    single_result_dir=results_dir+f'/{result_folder}'
    print(single_result_dir)
    for model_folder in os.listdir(single_result_dir):
        results_data['model'].append(model_folder)
        results_data['time'].append(result_folder)
        results_data['path_to_mae_img'].append(single_result_dir+f'/{model_folder}/listened_to_album_mae_plot.png')
        results_data['path_to_rmse_img'].append(single_result_dir+f'/{model_folder}/listened_to_album_rmse_plot.png')
        results_data['path_to_loss_img'].append(single_result_dir+f'/{model_folder}/listened_to_album_loss_plot.png')
        results_data['path_to_pkl'].append(single_result_dir+f'/{model_folder}/listened_to_album.pkl')
        # Opening metrics
        metrics = json.load(open(single_result_dir+f'/{model_folder}/metrics.json'))
        results_data['RMSE'].append(metrics['RMSE'])
        results_data['MAE'].append(metrics['MAE'])
        # Opening metrics
        args = json.load(open(single_result_dir+f'/{model_folder}/args.json'))
        results_data['nrows'].append(args['nrows'])
        results_data['num_layers'].append(args['num_layers'])
        results_data['sample_edge_rate'].append(args['sample_edge_rate'])
        results_data['batch_size'].append(args['batch_size'])
        results_data['num_neg_samples'].append(args['num_neg_samples'])
        results_data['node_min_neighbors'].append(args['node_min_neighbors'])
        results_data['shuffle'].append(args['shuffle'])
        results_data['drop_last'].append(args['drop_last'])
        results_data['num_workers'].append(args['num_workers'])
        results_data['hidden_dim'].append(args['hidden_dim'])
        results_data['rel_input_dim'].append(args['rel_input_dim'])  
        results_data['rel_hidden_dim'].append(args['rel_hidden_dim'])  
        results_data['num_heads'].append(args['num_heads'])
        results_data['dropout'].append(args['dropout'])
        results_data['residual'].append(args['residual'])
        results_data['norm'].append(args['norm'])
        results_data['opt'].append(args['opt'])
        results_data['learing_rate'].append(args['learing_rate'])
        results_data['weight_decay'].append(args['weight_decay'])
        results_data['epochs'].append(args['epochs'])
        results_data['device'].append(args['device'])
        results_data['patience'].append(args['patience'])
        results_data['overwrite_raw'].append(args['overwrite_raw'])
        results_data['overwrite_preprocessed'].append(args['overwrite_preprocessed'])
        results_data['overwrite_processed'].append(args['overwrite_processed'])



# for key in results_data.keys():
#     print(key, len(results_data[key]))
date=datetime.now().strftime("%d_%m_%Y_%H:%M:%S").replace(' ',"_")
os.mkdir(f'results/tuned_results/{date}')     
pd.DataFrame(results_data).to_csv(f'results/tuned_results/{date}/grid_search.csv')

save_args_path = f'results/tuned_results/{date}/args_ranges.json'
with open(save_args_path, 'w') as file:
    file.write(json.dumps(args_range_dict))
    file.close()
notes_path=f'results/tuned_results/{date}/notes.txt'
with open(notes_path, 'w') as file:
    file.write('Hello World!')
    file.close()
        
