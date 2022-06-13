import os

import torch as th
from tqdm import tqdm

from dgl import heterograph
from dgl.data import DGLDataset, download, extract_archive
from dgl.data.utils import save_graphs, load_graphs

from utils import CategoricalEncoder, IdentityEncoder, SequenceEncoder, add_reverse_edges, load_txt_df, mapIds, preprocess_listen_events_df, preprocess_raw
tqdm.pandas(desc="progress-bar")



class LFM1b(DGLDataset):
    def __init__(self, name='LFM-1b', hash_key=(), force_reload=False, verbose=False):
        self.root_dir = '../data/'+name
        self.preprocessed_dir = '../data/'+name+'/preprocessed'
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
            zip_file_path = os.path.join(self.root_dir, self.name+'.zip')
            extract_archive(download(self.url, path=zip_file_path, overwrite=False), self.root_dir, overwrite=False)

            zip_file_path = os.path.join(self.root_dir, self.name+'_UGP.zip')
            extract_archive(download('http://www.cp.jku.at/datasets/LFM-1b/LFM-1b_UGP.zip', path=zip_file_path, overwrite=False), self.root_dir, overwrite=False)              
                
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
            print(self.graph)

    def process(self):
        processed_condition = os.path.exists(os.path.join(self.save_dir+'/'+'lastfm1b.bin')) == False 
        if processed_condition == True:
            preprocess_raw(self.raw_dir,self.preprocessed_dir, nrows=1000000)
            graph_data = {}
            edge_data_features = {}
            node_data_features = {}
            # device = th.device("cuda" if th.cuda.is_available() else "cpu")
            device = th.device("cpu")
            mappings={}
            for filename in ['LFM-1b_artists.txt', 'LFM-1b_albums.txt', 'LFM-1b_tracks.txt', 'LFM-1b_users.txt', 'LFM-1b_LEs.txt']:
                file_path=self.preprocessed_dir+'/'+filename
                print('\t','-------------------',file_path.split('_')[-1],'-------------------')
                if filename=='LFM-1b_artists.txt':
                    df = load_txt_df(file_path, type='artist')
                    # -------------------------ARTIST ID MAPPING-------------------------
                    mappings['artist_mapping'] = {int(id): i for i, id in enumerate(df['artist_id'])}
                    # -------------------------ARTIST DF RE-ID-------------------------
                    df = mapIds(df, cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    # -------------------------ARTIST NODE FEATURES-------------------------
                    # artist node features
                    # encoders={'artist_name': SequenceEncoder(device=device),}
                    # node_data_features['artist'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = IdentityEncoder(dtype=th.float,device=device)
                    node_data_features['artist'] = {'feat': id_encoder(df['artist_id'].values)}
                    # del df
                elif filename=='LFM-1b_albums.txt':
                    df = load_txt_df(file_path, type='album')
                    # -------------------------ALBUM ID MAPPING-------------------------
                    mappings['album_mapping'] = {int(id): i for i, id in enumerate(df['album_id'])}
                    # -------------------------ALBUM DF RE-ID-------------------------
                    df = mapIds(df, cols=['artist_id', 'album_id'], mappings=[mappings['artist_mapping'], mappings['album_mapping']])
                    # -------------------------ALBUM NODE FEATURES-------------------------
                    # album node featuresd
                    # encoders={'album_name': SequenceEncoder(device=device),}
                    # node_data_features['album'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = IdentityEncoder(dtype=th.float,device=device)
                    node_data_features['album'] = {'feat': id_encoder(df['album_id'].values)}
                    # # -------------------------ALBUM->ARTIST EDGES-------------------------
                    # # album -> artist edge_list
                    graph_data[('album', 'produced_by', 'artist')]=(th.tensor(df['album_id'].values), th.tensor(df['artist_id'].values))
                    # del df
                elif filename=='LFM-1b_tracks.txt':
                    df = load_txt_df(file_path, type='track')
                    # -------------------------TRACK ID MAPPING-------------------------
                    mappings['track_mapping'] = {int(id): i for i, id in enumerate(df['track_id'])}
                    # -------------------------TRACK DF RE-ID-------------------------
                    df = mapIds(df, cols=['artist_id','track_id'], mappings=[mappings['artist_mapping'],mappings['track_mapping']])
                    # -------------------------TRACK NODE FEATURES-------------------------
                    # track node features
                    # encoders={'track_name': SequenceEncoder(device=device),}
                    # node_data_features['track'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = IdentityEncoder(dtype=th.float,device=device)
                    node_data_features['track'] = {'feat': id_encoder(df['track_id'].values)}
                    # -------------------------TRACK->ARTIST EDGES-------------------------
                    # track -> artist edge_list
                    graph_data[('track', 'preformed_by', 'artist')]=(th.tensor(df['track_id'].values), th.tensor(df['artist_id'].values))
                    # del df

                elif filename=='LFM-1b_users.txt':
                    df = load_txt_df(file_path, type='user')
                    # -------------------------USER ID MAPPING-------------------------
                    mappings['user_mapping']= {int(id): i for i, id in enumerate(df['user_id'])}
                    # -------------------------USER DF RE-ID-------------------------
                    df = mapIds(df, cols=['user_id'], mappings=[mappings['user_mapping']])
                    # -------------------------USER NODE FEATURES-------------------------
                    # user node features
                    # encoders={
                    #     'country': CategoricalEncoder(device=device),
                    #     'age': IdentityEncoder(dtype=th.long,device=device),
                    #     'gender': CategoricalEncoder(device=device),
                    #     'playcount': IdentityEncoder(dtype=th.float,device=device),
                    #     'registered_unixtime' : IdentityEncoder(dtype=th.long,device=device),
                    # }
                    # node_data_features['user'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = IdentityEncoder(dtype=th.float,device=device)
                    node_data_features['user'] = {'feat': id_encoder(df['user_id'].values)}
                    # del df

                elif filename=='LFM-1b_LEs.txt':
                    df = load_txt_df(file_path, type='le')
                    encoders={
                        # 'playcount': IdentityEncoder(),
                        # 'last_played' : IdentityEncoder(),
                    }
                    # -------------------------LE DF RE-ID-------------------------
                    # df = mapIds(df, cols=['user_id', 'artist_id', 'album_id', 'track_id'], mappings=[user_mapping, artist_mapping, album_mapping, track_mapping])
                    df = mapIds(df, cols=['user_id', 'artist_id', 'album_id', 'track_id'], mappings=[mappings['user_mapping'], mappings['artist_mapping'], mappings['album_mapping'], mappings['track_mapping']])
                    # -------------------------USER->ALBUMS-------------------------
                    le_df = preprocess_listen_events_df(df, type='album')
                    # user -> album edge_list
                    graph_data[('user', 'listened_to_album', 'album')]=(th.tensor(le_df['user_id'].values), th.tensor(le_df['album_id'].values))
                    # user -> album features
                    edge_data_features[('user', 'listened_to_album', 'album')] = {col:encoder(le_df[col].values) for col, encoder in encoders.items()}

                    # -------------------------USER->ARTISTS-------------------------
                    le_df = preprocess_listen_events_df(df, type='artist')
                    # user -> artist edge_list
                    graph_data[('user', 'listened_to_artist', 'artist')]=(th.tensor(le_df['user_id'].values), th.tensor(le_df['artist_id'].values))
                    # user -> artist features
                    edge_data_features[('user', 'listened_to_artist', 'artist')] = {col:encoder(le_df[col].values) for col, encoder in encoders.items()}
                    
                    # -------------------------USER->TRACKS-------------------------
                    le_df = preprocess_listen_events_df(df, type='track')
                    # user -> artist edge_list
                    graph_data[('user', 'listened_to_track', 'track')]=(th.tensor(le_df['user_id'].values), th.tensor(le_df['track_id'].values))
                    # user -> artist features
                    edge_data_features[('user', 'listened_to_track', 'track')] = {col:encoder(le_df[col].values) for col, encoder in encoders.items()}
                    
                    del le_df
                else:
                    raise Exception('filename in processed directory is bad.. Filename:',filename)
            
                del df

            # self.raw_UGP_dir=self.raw_dir+'_UGP'
            # for filename in ['LFM-1b_artists.txt', 'LFM-1b_albums.txt', 'LFM-1b_tracks.txt', 'LFM-1b_users.txt', 'LFM-1b_LEs.txt']:
            #     file_path=self.preprocessed_dir+'/'+filename
            #     print('\t','-------------------',file_path.split('_')[-1],'-------------------')

            self.graph = heterograph(graph_data)
            
            for edge in edge_data_features:
                for feature in edge_data_features[edge].keys():
                    feature_data = edge_data_features[edge][feature]
                    self.graph.edges[edge].data[feature] = feature_data
            del edge_data_features

            for node in node_data_features:
                for feature in node_data_features[node].keys():
                    feature_data = node_data_features[node][feature]
                    self.graph.nodes[node].data[feature] = feature_data
            del node_data_features

            self.graph = add_reverse_edges(self.graph)

    def __getitem__(self, idx):
        glist,_=self.load()
        return glist[idx]

    def __len__(self):
        return 1
 