from datetime import datetime
import itertools
from train import train_models
import os
import numpy
import json
import pandas as pd

# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

args_range_dict={
    'seed':[0],
    'n_users':[None],
    'num_layers':[2,6],
    'sample_edge_rate':[0.01,.05],
    'batch_size':[512,1024],
    'num_neg_samples':[2, 5],
    'node_min_neighbors':[2, 5],
    'shuffle':[True],
    'drop_last':[False],
    'num_workers':[4],
    'hidden_dim':[32,64],
    'rel_input_dim':[6,12],
    'rel_hidden_dim':[24, 32],
    'num_heads':[8],
    'dropout':[0.3,.05,.08],
    'residual':[True, False],
    'norm':[True],
    'opt':['adam'],
    'learing_rate':[.001, .0001],
    'weight_decay':[0.0, 0.1],
    'epochs':[150],
    'device':['cuda'],
    'patience':[25],
    'overwrite_raw':[False],
    'overwrite_preprocessed':[False],
    'overwrite_processed':[False]
}

print('Tuning Hyperparameters')

keys, values = zip(*args_range_dict.items())
permutations_args = [dict(zip(keys, v)) for v in itertools.product(*values)]
for index, arg_perm in enumerate(permutations_args):
    print(f'Tuning Hyperparameters permutation #{index} of {len(permutations_args)}')
    # train_models(Dict2Class(arg_perm))
    args = Dict2Class(arg_perm)
    train_models(args)

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
        