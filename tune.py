
from datetime import datetime
import itertools
import json
import os
import pandas as pd
import torch as th

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
    'sample_edge_rate':[.01],
    'num_layers':[2],
    'batch_size':[512],
    'num_neg_samples': [10],
    'node_min_neighbors':[10],
    'shuffle':[True],
    'drop_last':[False],
    'num_workers':[4],
    'hidden_dim':[16],
    'rel_input_dim':[8],
    'rel_hidden_dim':[16],
    'num_heads':[8],
    'dropout':[.5,],
    'residual':[True],
    'norm':[True],
    'opt':['adam'],
    'weight_decay':[0.0],
    'epochs':[100],
    'patience':[10],
    'split_by_users':[True],
    'device':['cuda'],
    'artists':[True, False],
    'albums':[True, False],
    'tracks':[True, False],
    'playcount_weight':[False],
    'norm_playcount_weight':[False],
    'metapath2vec':[True],
    'emb_dim':[8],
    'walk_length':[64],
    'context_size':[7],
    'walks_per_node':[1],
    'metapath2vec_epochs_batch_size':[512],
    'learning_rate':[0.001],
    'metapath2vec_epochs':[5],
    'logs':[100],
    'n_users':[10],
    'popular_artists':[True],

}



print('\n')
print('Tuning Hyperparameters')

keys, values = zip(*args_range_dict.items())
permutations_args = [dict(zip(keys, v)) for v in itertools.product(*values)]

for user_size in args_range_dict['n_users']:
    for i, arg_perm in enumerate([perm for perm in permutations_args if Dict2Class(perm).n_users==user_size]):
        print('running with args')
        print(arg_perm)
        args=Dict2Class(arg_perm)
        
        try:
            if (i==0):
                print('OVERWRITING PREPROCESSED')
                dataset = LFM1b(
                n_users=args.n_users, 
                popular_artists=args.popular_artists,
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
                num_negative_samples=args.num_neg_samples,
                batch_size=args.metapath2vec_epochs_batch_size,
                learning_rate=args.learning_rate,
                epochs=args.metapath2vec_epochs,
                logs=args.logs
                )

            else:
                'OVERWRITING_PROCESSED'
                dataset = LFM1b(
                n_users=args.n_users, 
                popular_artists=args.popular_artists,
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
                num_negative_samples=args.num_neg_samples,
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
        except Exception as e:
            print(e)



results_dir='results/lfm1b'
results_data={
    'model':[],
    'model_parameters':[],
    'time':[],
    'path_to_loss_img':[],
    'path_to_mae_img':[],
    'path_to_rmse_img':[],
    'path_to_auc_img':[],
    'path_to_ap_img':[],
    'path_to_hit_img':[],
    'path_to_div_img':[],
    'path_to_cov_img':[],
    'path_to_pkl':[],
    'RMSE':[],
    'MAE':[],
    'AUC':[],
    'AP':[],
    'HIT':[],
    'DIV':[], 
    'COV':[],   
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
    'walks_per_node':[],
    'metapath2vec_epochs_batch_size':[],
    'learning_rate':[],
    'metapath2vec_epochs':[],
    'logs':[],
    'n_users':[],
    'popular_artists':[],
}
for result_folder in os.listdir(results_dir):
    single_result_dir=results_dir+f'/{result_folder}'
    for model_folder in os.listdir(single_result_dir):
        try:
            metrics = json.load(open(single_result_dir+f'/{model_folder}/metrics.json'))
            params=['AUC','AP','RMSE','MAE','HIT','DIV','COV']
            for param in params:
                results_data[param].append(metrics[param])

            rec_metrics = json.load(open(single_result_dir+f'/{model_folder}/rec_metrics.json'))
            params=['avg_artist_dist','avg_album_dist','avg_track_dist']
            for param in params:
                results_data[param].append(metrics[param])
            
            args = json.load(open(single_result_dir+f'/{model_folder}/args.json'))
            params=[
                    'seed',
                    'sample_edge_rate',
                    'num_layers',
                    'batch_size',
                    'num_neg_samples',
                    'node_min_neighbors',
                    'shuffle',
                    'drop_last',
                    'num_workers',
                    'hidden_dim',
                    'rel_input_dim',
                    'rel_hidden_dim',
                    'num_heads',
                    'dropout',
                    'residual',
                    'norm',
                    'opt',
                    'weight_decay',
                    'epochs',
                    'patience',
                    'split_by_users',
                    'device',
                    'artists',
                    'albums',
                    'tracks',
                    'playcount_weight',
                    'norm_playcount_weight',
                    'metapath2vec',
                    'emb_dim',
                    'walk_length',
                    'context_size',
                    'walks_per_node',
                    'metapath2vec_epochs_batch_size',
                    'learning_rate',
                    'metapath2vec_epochs',
                    'logs',
                    'n_users',
                    'popular_artists',
                    'model_parameters'
                    ]
            for param in params:
                results_data[param].append(args[param])
            results_data['model'].append(model_folder)
            results_data['time'].append(result_folder)
            results_data['path_to_mae_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_mae_plot.png')
            results_data['path_to_rmse_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_rmse_plot.png')
            results_data['path_to_loss_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_loss_plot.png')
            results_data['path_to_auc_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_auc_plot.png')
            results_data['path_to_ap_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_ap_plot.png')
            results_data['path_to_hit_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_hit_plot.png')
            results_data['path_to_div_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_div_plot.png')
            results_data['path_to_cov_img'].append(single_result_dir+f'/{model_folder}/{model_folder}_cov_plot.png')
            results_data['path_to_pkl'].append(single_result_dir+f'/{model_folder}/{model_folder}.pkl')
        except Exception as e:
            print(e)
for k,v in results_data.items():
    print(k,len(v))

date=datetime.now().strftime("%d_%m_%Y_%H:%M:%S").replace(' ',"_")
if os.path.exists(f'results/tuned_results')==False:
    os.mkdir(f'results/tuned_results')
os.mkdir(f'results/tuned_results/{date}')     
pd.DataFrame(results_data).to_csv(f'results/tuned_results/{date}/grid_search.csv')

notes_path=f'results/tuned_results/{date}/notes.txt'
with open(notes_path, 'w') as file:
    file.write('Hello World!')
    file.close()