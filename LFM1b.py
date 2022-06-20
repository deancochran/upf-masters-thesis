from logging import raiseExceptions
import os
import torch as th
from dgl import heterograph
from dgl.data import DGLDataset, download, extract_archive
from dgl.data.utils import save_graphs, load_graphs
from data_utils import get_artist_genre_df, get_full_le_plays, get_le_playcount, get_preprocessed_ids, remap_ids
from preprocess import preprocess_raw
from encoders import CategoricalEncoder, IdentityEncoder


class LFM1b(DGLDataset):
    def __init__(self, name='LFM-1b', hash_key=(), force_reload=False, verbose=False, nrows=None, overwrite_processed=False, overwrite_preprocessed=False, overwrite_raw=False):
        self.root_dir = 'data/'+name
        self.preprocessed_dir = 'data/'+name+'/preprocessed'
        self.nrows=nrows
        self.overwrite_processed=overwrite_processed
        self.overwrite_preprocessed=overwrite_preprocessed
        self.overwrite_raw=overwrite_raw

        self.lfm1b_ugp_url='http://www.cp.jku.at/datasets/LFM-1b/LFM-1b_UGP.zip'
        self.raw_ugp_dir='data/'+name+f'/{name}_UGP'
        super().__init__(
            name=name, 
            url='http://drive.jku.at/ssf/s/readFile/share/1056/266403063659030189/publicLink/LFM-1b.zip', 
            raw_dir=self.root_dir+'/'+name, 
            save_dir=self.root_dir+'/processed',
            hash_key=hash_key, 
            force_reload=force_reload, 
            verbose=verbose
            ) 

    def download(self):
        """Download and extract Zip file from LFM1b"""
        if self.url is not None:
            extract_archive(download(self.url, path = self.root_dir, overwrite = False), target_dir = self.root_dir+'/'+self.name, overwrite = self.overwrite_raw)
            # LMF-1b_UGP.zip download goes here if needed
            extract_archive(download(self.lfm1b_ugp_url, path = self.root_dir, overwrite = False), target_dir = self.root_dir, overwrite = self.overwrite_raw)
            # extract_archive(download(lfm1b_ugp_zip_url, path=self.root_dir, overwrite=False), self.root_dir, overwrite=False)
            
            if not os.path.exists(self.preprocessed_dir):
                os.mkdir(self.preprocessed_dir)     
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)              
            
                
        else:
            raise Exception("self.url is None! This should point to the LastFM1b zip download path: 'http://drive.jku.at/ssf/s/readFile/share/1056/266403063659030189/publicLink/LFM-1b.zip'")

    def load(self):
        """load graph list and graph labels with load_graphs"""
        return load_graphs(self.save_dir+'/lastfm1b.bin')

    def save(self):
        """save file to processed directory"""
        if os.path.exists(os.path.join(self.save_dir+'/lastfm1b.bin')) == False:
            glist=[self.graph]
            glabels={"glabel": th.tensor([0])}
            save_graphs(self.save_dir+'/lastfm1b.bin',glist,glabels)

    # processes raw files into the preprocessed directory
    def process(self):
        processed_condition = os.path.exists(os.path.join(self.save_dir+'/'+'lastfm1b.bin')) == False
        if processed_condition == True or self.overwrite_processed==True or self.overwrite_preprocessed==True:
            preprocess_raw(self.raw_dir,self.preprocessed_dir, nrows=self.nrows, overwrite=self.overwrite_preprocessed)
            graph_data = {}
            edge_data_features = {}
            node_data_features = {}
            # device = th.device("cuda" if th.cuda.is_available() else "cpu") # this dataset only can be processed on cpu
            device = th.device("cpu")
            mappings={} # a id remapping for all the ids in the data base
            list_of_filenames=['LFM-1b_artists.txt', 'genres_allmusic.txt', 'LFM-1b_albums.txt', 'LFM-1b_tracks.txt', 'LFM-1b_users.txt', 'LFM-1b_LEs.txt']
            for filename in list_of_filenames:
                file_path=self.preprocessed_dir+'/'+filename
                print('\t','------------------- Loading Info from',file_path.split('_')[-1],'-------------------')

                id_encoder = IdentityEncoder(dtype=th.float,device=device) # used to encode floats f(x)==2, where x = 2
                cat_encoder = CategoricalEncoder(device=device) # used to encode categories f(x)==[0,0,1,0], where x = 2, of possible types 0,1,2,3
                # seq_encoder = SequenceEncoder(device=device) # used to encode strs f(x)==[0.213,0.254,...,134,.893], where x = 'dean', and shape is (1x254)
                
                if filename=='LFM-1b_artists.txt':
                    # -------------------------ARTIST ID RE-MAPPING-------------------------
                    df = get_preprocessed_ids(file_path, type='artist', id_list=['artist_id','artist_name'])
                    mappings['artist_mapping'] = {int(id): i for i, id in enumerate(df['artist_id'])}
                    df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    mappings['artist_name_mapping'] = {artist_name: int(artist_id)  for artist_id, artist_name in zip(df['artist_id'],df['artist_name'])}
                    # -------------------------ARTIST NODE FEATURES-------------------------
                    node_data_features['artist'] = {'feat': cat_encoder(df['artist_id'])}

                    del df
                    

                elif filename=='genres_allmusic.txt':
                    df = get_preprocessed_ids(file_path,type='genre', return_unique_ids=True ,id_list=['genre_id'])
                    # -------------------------GENRE NODE FEATURES-------------------------
                    node_data_features['genre'] = {'feat': cat_encoder(df['genre_id'])}
                    del df

                    artist_genres_path=self.raw_ugp_dir+'/LFM-1b_artist_genres_allmusic.txt'
                    df = get_artist_genre_df(artist_genres_path, mappings['artist_name_mapping'])
                    # -------------------------ARTIST->GENRE EDGES-------------------------
                    graph_data[('artist', 'in_genre', 'genre')]=(th.tensor(df['artist_id'].values), th.tensor(df['genre_id'].values))
                    edge_data_features['in_genre']={'norm_playcount': id_encoder([1 for id in df['genre_id']])}
                    # -------------------------GENRE->ARTIST EDGES-------------------------
                    graph_data[('genre', 'is_genre_of', 'artist')]=(th.tensor(df['genre_id'].values), th.tensor(df['artist_id'].values))
                    edge_data_features['is_genre_of']={'norm_playcount': id_encoder([1 for id in df['genre_id']])}

                    del df
                    del mappings['artist_name_mapping']

                
                
                elif filename=='LFM-1b_albums.txt':
                    # -------------------------ALBUM ID RE-MAPPING-------------------------
                    df = get_preprocessed_ids(file_path, type='album', id_list=['album_id','artist_id'])
                    mappings['album_mapping'] = {int(id): i for i, id in enumerate(df['album_id'])}
                    df=remap_ids(df, ordered_cols=['artist_id','album_id'], mappings=[mappings['artist_mapping'], mappings['album_mapping']])
                    # -------------------------ALBUM NODE FEATURES-------------------------
                    node_data_features['album'] = {'feat': cat_encoder(df['album_id'])}
                    # -------------------------ALBUM->ARTIST EDGES-------------------------
                    graph_data[('album', 'produced_by', 'artist')]=(th.tensor(df['album_id']), th.tensor(df['artist_id']))
                    edge_data_features['produced_by']={'norm_playcount': id_encoder([1 for id in df['album_id']])}
                    # -------------------------ARTIST->ALBUM EDGES-------------------------
                    graph_data[('artist', 'produced', 'album')]=(th.tensor(df['artist_id']), th.tensor(df['album_id']))
                    edge_data_features['produced']={'norm_playcount': id_encoder([1 for id in df['album_id']])}

                    del df


                elif filename=='LFM-1b_tracks.txt':
                    # -------------------------TRACK ID RE-MAPPING-------------------------
                    df = get_preprocessed_ids(file_path, type='track', id_list=['track_id','artist_id'])
                    mappings['track_mapping'] = {int(id): i for i, id in enumerate(df['track_id'])}
                    df=remap_ids(df, ordered_cols=['artist_id','track_id'], mappings=[mappings['artist_mapping'], mappings['track_mapping']])
                    # -------------------------TRACK NODE FEATURES-------------------------
                    node_data_features['track'] = {'feat': cat_encoder(df['track_id'])}
                    # -------------------------TRACK->ARTIST EDGES-------------------------
                    graph_data[('track', 'preformed_by', 'artist')]=(th.tensor(df['track_id']), th.tensor(df['artist_id']))
                    edge_data_features['preformed_by']={'norm_playcount': id_encoder([1 for id in df['track_id']])}
                    # -------------------------ARTIST->TRACK EDGES-------------------------
                    graph_data[('artist', 'preformed', 'track')]=(th.tensor(df['artist_id']), th.tensor(df['track_id']))
                    edge_data_features['preformed']={'norm_playcount': id_encoder([1 for id in df['track_id']])}

                    del df


                elif filename=='LFM-1b_users.txt':
                    # -------------------------USER ID RE-MAPPING-------------------------
                    df = get_preprocessed_ids(file_path, type='user', id_list=['user_id'])
                    mappings['user_mapping']= {int(id): i for i, id in enumerate(df['user_id'])}
                    df=remap_ids(df, ordered_cols=['user_id'], mappings=[mappings['user_mapping']])
                    # -------------------------USER NODE FEATURES-------------------------
                    node_data_features['user'] = {'feat': cat_encoder(df['user_id'])}
                    del df

                elif filename=='LFM-1b_LEs.txt':
                    # -------------------------USER->ARTISTS-------------------------
                    user_id_list, groupby_id_list, playcounts=get_le_playcount(
                        file_path,type_id='artist_id',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['artist_mapping'], 
                        relative_playcount=True
                        )
                    graph_data[('user', 'listened_to_artist', 'artist')]=(
                        th.tensor(user_id_list), 
                        th.tensor(groupby_id_list)
                        )
                    edge_data_features['listened_to_artist']={'norm_playcount': id_encoder(playcounts)}

                    graph_data[('artist', 'artist_listened_by', 'user')]=(
                        th.tensor(groupby_id_list), 
                        th.tensor(user_id_list)
                        )
                    edge_data_features['artist_listened_by']={'norm_playcount': id_encoder(playcounts)}
                    del mappings['artist_mapping']
                    del user_id_list
                    del groupby_id_list
                    del playcounts

                    # -------------------------USER->ALBUMS-------------------------                    
                    user_id_list, groupby_id_list, playcounts=get_le_playcount(
                        file_path,type_id='album_id',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['album_mapping'], 
                        relative_playcount=True
                        )
                    graph_data[('user', 'listened_to_album', 'album')]=(
                        th.tensor(user_id_list), 
                        th.tensor(groupby_id_list)
                        )
                    edge_data_features['listened_to_album']={'norm_playcount': id_encoder(playcounts)}

                    graph_data[('album', 'album_listened_by', 'user')]=(
                        th.tensor(groupby_id_list), 
                        th.tensor(user_id_list)
                        )
                    edge_data_features['album_listened_by']={'norm_playcount': id_encoder(playcounts)}
                    del mappings['album_mapping']
                    del user_id_list
                    del groupby_id_list
                    del playcounts
                    
                    # -------------------------USER->TRACKS-------------------------
                    user_id_list, groupby_id_list, playcounts=get_le_playcount(
                        file_path,type_id='track_id',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['track_mapping'], 
                        relative_playcount=True
                        )
                    graph_data[('user', 'listened_to_track', 'track')]=(
                        th.tensor(user_id_list), 
                        th.tensor(groupby_id_list)
                        )
                    edge_data_features['listened_to_track']={'norm_playcount': id_encoder(playcounts)}

                    graph_data[('track', 'track_listened_by', 'user')]=(
                        th.tensor(groupby_id_list), 
                        th.tensor(user_id_list)
                        )
                    edge_data_features['track_listened_by']={'norm_playcount': id_encoder(playcounts)}
                    del mappings['track_mapping']
                    del user_id_list
                    del groupby_id_list
                    del playcounts

                else:
                    raise Exception('filename in processed directory is bad.. Filename:',filename)
                
            del mappings

            print('\t','-------------------  Creating Graph from data  -------------------')

            # create graph data
            self.graph = heterograph(graph_data)
            # print(self.graph)
            del graph_data
            
            print('\t','-------------------  Setting Graph edge data  -------------------')
            # init graph edge data
            for edge in edge_data_features:
                for feature in edge_data_features[edge].keys():
                    feature_data = edge_data_features[edge][feature]
                    print(f'assigning feature of shape {feature_data.shape} to {edge}')
                    self.graph.edges[edge].data[feature] = feature_data
            del edge_data_features

            print('\t','-------------------  Setting Graph node data  -------------------')
            # init graph node data
            for node in node_data_features.keys():
                for feature in node_data_features[node].keys():
                    feature_data = node_data_features[node][feature]
                    print(f'assigning feature of shape {feature_data.shape} to {node}')
                    self.graph.nodes[node].data[feature] = feature_data
            del node_data_features

    def __getitem__(self, idx):
        glist,_=self.load()
        return glist[idx]

    def __len__(self):
        return 1
  