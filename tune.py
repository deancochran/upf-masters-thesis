from distutils.log import error
import itertools
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
    'sample_edge_rate':[0.2,0.01],
    'num_layers':[2,4],
    'batch_size':[512],
    'num_neg_samples': [10],
    'node_min_neighbors':[10],
    'shuffle':[True],
    'drop_last':[False],
    'num_workers':[4],
    'hidden_dim':[16,64],
    'rel_input_dim':[12],
    'rel_hidden_dim':[16,64],
    'num_heads':[8],
    'dropout':[.5],
    'residual':[True],
    'norm':[True],
    'opt':['adam'],
    'learing_rate':[.001],
    'weight_decay':[0.0],
    'epochs':[75],
    'patience':[25],
    'split_by_users':[True,False],
    'device':['cuda'],
    'artists':[True, False],
    'albums':[True, False],
    'tracks':[True, False],
    'playcount_weight':[True,False],
    'norm_playcount_weight':[True,False],
    'metapath2vec':[False],
    'emb_dim':[128],
    'walk_length':[64],
    'context_size':[7],
    'walks_per_node':[3],
    'num_negative_samples':[3],
    'metapath2vec_epochs_batch_size':[128],
    'learning_rate':[0.01],
    'metapath2vec_epochs':[5],
    'logs':[100],


    'n_users':[10,50,100],


}



print('\n')
print('Tuning Hyperparameters')

keys, values = zip(*args_range_dict.items())
permutations_args = [dict(zip(keys, v)) for v in itertools.product(*values)]

for n_users in args_range_dict['n_users']:
    for i, arg_perm in enumerate(permutations_args):
        args=Dict2Class(arg_perm)
        if args.n_users == n_users:
            if (i==0):
                dataset = LFM1b(
                n_users=args.n_users, 
                device=args.device, 
                overwrite_preprocessed=True,
                overwrite_processed=True,
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
                num_negative_samples=args.num_negative_samples,
                batch_size=args.metapath2vec_epochs_batch_size,
                learning_rate=args.learning_rate,
                epochs=args.metapath2vec_epochs,
                logs=args.logs
                )
                copy_args=args.copy()
            elif (copy_args.artists != args.artists) or (copy_args.albums != args.albums) or (copy_args.tracks != args.tracks) or (copy_args.playcount_weight != args.playcount_weight) or (copy_args.norm_playcount_weight != args.norm_playcount_weight):
                dataset = LFM1b(
                n_users=args.n_users, 
                device=args.device, 
                overwrite_preprocessed=False,
                overwrite_processed=True,
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
                num_negative_samples=args.num_negative_samples,
                batch_size=args.metapath2vec_epochs_batch_size,
                learning_rate=args.learning_rate,
                epochs=args.metapath2vec_epochs,
                logs=args.logs
                )
                copy_args=args.copy()
            else:
                dataset = LFM1b(
                n_users=args.n_users, 
                device=args.device, 
                overwrite_preprocessed=False,
                overwrite_processed=False,
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
                num_negative_samples=args.num_negative_samples,
                batch_size=args.metapath2vec_epochs_batch_size,
                learning_rate=args.learning_rate,
                epochs=args.metapath2vec_epochs,
                logs=args.logs
                )
                
            print('Loading graph')
            glist,_= dataset.load()
            hg=glist[0] 
            try:
                train_models(hg, args)
            except Exception as e:
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
    'seed':[],
    'sample_edge_rate':[],
    'num_layers':[],
    'batch_size':[],
    'num_neg_samples': [],
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
    'patience':[],
    'split_by_users':[],
    'device':[],
    'artists':[],
    'albums':[],
    'tracks':[],
    'playcount_weight':[],
    'norm_playcount_weight':[],
    'metapath2vec':[],
    'emb_dim':[],
    'walk_length':[],
    'context_size':[],
    'num_negative_samples':[],
    'metapath2vec_epochs_batch_size':[],
    'learning_rate':[],
    'metapath2vec_epochs':[],
    'logs':[],
    'n_users':[],
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
        results_data['seed'].append(args['seed'])
        results_data['n_users'].append(args['n_users'])
        results_data['sample_edge_rate'].append(args['sample_edge_rate'])
        results_data['num_layers'].append(args['num_layers'])
        results_data['batch_size'].append(args['batch_size'])
        results_data['num_neg_samples'].append(args['num_neg_samples'])
        results_data['shuffle'].append(args['shuffle'])
        results_data['drop_last'].append(args['drop_last'])
        results_data['num_workers'].append(args['num_workers'])
        results_data['hidden_dim'].append(args['hidden_dim'])
        results_data['rel_input_dim'].append(args['rel_input_dim'])
        results_data['rel_hidden_dim'].append(args['rel_hidden_dim'])
        results_data['num_heads'].append(args['num_heads'])
        results_data['dropout'].append(args['dropout'])
        results_data['dropout'].append(args['dropout'])
        results_data['dropout'].append(args['dropout'])
        results_data['dropout'].append(args['dropout'])
        results_data['dropout'].append(args['dropout'])
        results_data['residual'].append(args['residual'])
        results_data['opt'].append(args['opt'])
        results_data['learing_rate'].append(args['learing_rate'])
        results_data['weight_decay'].append(args['weight_decay'])
        results_data['epochs'].append(args['epochs'])
        results_data['patience'].append(args['patience'])
        results_data['split_by_users'].append(args['split_by_users'])
        results_data['device'].append(args['device'])
        results_data['artists'].append(args['artists'])
        results_data['albums'].append(args['albums'])
        results_data['tracks'].append(args['tracks'])
        results_data['norm_playcount_weight'].append(args['norm_playcount_weight'])
        results_data['metapath2vec'].append(args['metapath2vec'])
        results_data['emb_dim'].append(args['emb_dim'])
        results_data['walk_length'].append(args['walk_length'])
        results_data['context_size'].append(args['context_size'])
        results_data['num_negative_samples'].append(args['num_negative_samples'])
        results_data['metapath2vec_epochs_batch_size'].append(args['metapath2vec_epochs_batch_size'])
        results_data['learning_rate'].append(args['learning_rate'])
        results_data['metapath2vec_epochs'].append(args['metapath2vec_epochs'])
        results_data['logs'].append(args['logs'])
        


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
        
