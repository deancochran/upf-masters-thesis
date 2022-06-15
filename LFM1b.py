import os
import torch as th
from dgl import heterograph
from dgl.data import DGLDataset, download, extract_archive
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm
from data_utils import CategoricalEncoder, IdentityEncoder, SequenceEncoder, add_reverse_edges, load_txt_df, mapIds, preprocess_listen_events_df, preprocess_raw, get_artist_genre_df

class LFM1b(DGLDataset):
    def __init__(self, name='LFM-1b', hash_key=(), force_reload=False, verbose=False, nrows=None, overwrite=False):
        self.root_dir = 'data/'+name
        self.preprocessed_dir = 'data/'+name+'/preprocessed'
        self.nrows=nrows
        self.overwrite=overwrite
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
            extract_archive(download(self.url, path = self.root_dir, overwrite = False), target_dir = self.root_dir+'/'+self.name, overwrite = self.overwrite)
            # LMF-1b_UGP.zip download goes here if needed
            extract_archive(download(self.lfm1b_ugp_url, path = self.root_dir, overwrite = False), target_dir = self.root_dir, overwrite = self.overwrite)
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
            print(self.graph)

    # processes raw files into the preprocessed directory
    def process(self):
        processed_condition = os.path.exists(os.path.join(self.save_dir+'/'+'lastfm1b.bin')) == False 
        if processed_condition == True:
            preprocess_raw(self.raw_dir,self.preprocessed_dir, nrows=self.nrows, overwrite=self.overwrite)
            graph_data = {}
            edge_data_features = {}
            node_data_features = {}
            # device = th.device("cuda" if th.cuda.is_available() else "cpu") # this dataset only can be processed on cpu
            device = th.device("cpu")
            mappings={} # a id remapping for all the ids in the data base
            for filename in ['genres_allmusic.txt', 'LFM-1b_artists.txt', 'LFM-1b_albums.txt', 'LFM-1b_tracks.txt', 'LFM-1b_users.txt', 'LFM-1b_LEs.txt']:
                file_path=self.preprocessed_dir+'/'+filename
                print('\t','-------------------',file_path.split('_')[-1],'-------------------')
                if filename=='genres_allmusic.txt':
                    df = load_txt_df(file_path, type='genre')
                    # -------------------------GENRE NODE FEATURES-------------------------
                    # encoders={'genre_name': SequenceEncoder(device=device),}
                    # node_data_features['genre'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = CategoricalEncoder(device=device)
                    node_data_features['genre'] = {'feat': id_encoder(df['genre_id'].values)}

                elif filename=='LFM-1b_artists.txt':
                    # -------------------------ARTIST ID RE-MAPPING-------------------------
                    df = load_txt_df(file_path, type='artist')
                    mappings['artist_mapping'] = {int(id): i for i, id in enumerate(df['artist_id'])}
                    df = mapIds(df, cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    # -------------------------ARTIST NODE FEATURES-------------------------
                    # encoders={'artist_name': SequenceEncoder(device=device),}
                    # node_data_features['artist'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = CategoricalEncoder(device=device)
                    node_data_features['artist'] = {'feat': id_encoder(df['artist_id'].values)}
                    # -------------------------ARTIST->GENRE EDGES-------------------------
                    mappings['artist_name_mapping'] = {row['artist_name']: row['artist_id']  for i, row in df.iterrows()}
                    df = get_artist_genre_df(self.raw_ugp_dir+'/LFM-1b_artist_genres_allmusic.txt', df['artist_name'].tolist(), mappings['artist_name_mapping'])
                    graph_data[('artist', 'in_genre', 'genre')]=(th.tensor(df['artist_id'].values), th.tensor(df['genre_id'].values))
                
                elif filename=='LFM-1b_albums.txt':
                    df = load_txt_df(file_path, type='album')
                    # -------------------------ALBUM ID RE-MAPPING-------------------------
                    mappings['album_mapping'] = {int(id): i for i, id in enumerate(df['album_id'])}
                    df = mapIds(df, cols=['artist_id', 'album_id'], mappings=[mappings['artist_mapping'], mappings['album_mapping']])
                    # -------------------------ALBUM NODE FEATURES-------------------------
                    # encoders={'album_name': SequenceEncoder(device=device),}
                    # node_data_features['album'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = CategoricalEncoder(device=device)
                    node_data_features['album'] = {'feat': id_encoder(df['album_id'].values)}
                    # -------------------------ALBUM->ARTIST EDGES-------------------------
                    graph_data[('album', 'produced_by', 'artist')]=(th.tensor(df['album_id'].values), th.tensor(df['artist_id'].values))

                elif filename=='LFM-1b_tracks.txt':
                    df = load_txt_df(file_path, type='track')
                    # -------------------------TRACK ID RE-MAPPING-------------------------
                    mappings['track_mapping'] = {int(id): i for i, id in enumerate(df['track_id'])}
                    df = mapIds(df, cols=['artist_id','track_id'], mappings=[mappings['artist_mapping'],mappings['track_mapping']])
                    # -------------------------TRACK NODE FEATURES-------------------------
                    # encoders={'track_name': SequenceEncoder(device=device),}
                    # node_data_features['track'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = CategoricalEncoder(device=device)
                    node_data_features['track'] = {'feat': id_encoder(df['track_id'].values)}
                    # -------------------------TRACK->ARTIST EDGES-------------------------
                    graph_data[('track', 'preformed_by', 'artist')]=(th.tensor(df['track_id'].values), th.tensor(df['artist_id'].values))

                elif filename=='LFM-1b_users.txt':
                    df = load_txt_df(file_path, type='user')
                    # -------------------------USER ID RE-MAPPING-------------------------
                    mappings['user_mapping']= {int(id): i for i, id in enumerate(df['user_id'])}
                    df = mapIds(df, cols=['user_id'], mappings=[mappings['user_mapping']])
                    # -------------------------USER NODE FEATURES-------------------------
                    # encoders={
                    #     'country': CategoricalEncoder(device=device),
                    #     'age': IdentityEncoder(dtype=th.long,device=device),
                    #     'gender': CategoricalEncoder(device=device),
                    #     'playcount': IdentityEncoder(dtype=th.float,device=device),
                    #     'registered_unixtime' : IdentityEncoder(dtype=th.long,device=device),
                    # }
                    # node_data_features['user'] = {col:encoder(df[col].values) for col, encoder in encoders.items()}
                    id_encoder = CategoricalEncoder(device=device)
                    node_data_features['user'] = {'feat': id_encoder(df['user_id'].values)}

                elif filename=='LFM-1b_LEs.txt':
                    df = load_txt_df(file_path, type='le')
                    encoders={
                        'playcount': IdentityEncoder(dtype=th.float,device=device),
                        # 'last_played' : IdentityEncoder(dtype=th.float,device=device),
                    }
                    # -------------------------LE IDs RE-MAPPING-------------------------
                    df = mapIds(df, cols=['user_id', 'artist_id', 'album_id', 'track_id'], mappings=[mappings['user_mapping'], mappings['artist_mapping'], mappings['album_mapping'], mappings['track_mapping']])
                    # -------------------------USER->ALBUMS-------------------------
                    le_df = preprocess_listen_events_df(df, type='album')
                    graph_data[('user', 'listened_to_album', 'album')]=(th.tensor(le_df['user_id'].values), th.tensor(le_df['album_id'].values))
                    edge_data_features[('user', 'listened_to_album', 'album')] = {col:encoder(le_df[col].values) for col, encoder in encoders.items()}

                    # -------------------------USER->ARTISTS-------------------------
                    le_df = preprocess_listen_events_df(df, type='artist')
                    graph_data[('user', 'listened_to_artist', 'artist')]=(th.tensor(le_df['user_id'].values), th.tensor(le_df['artist_id'].values))
                    edge_data_features[('user', 'listened_to_artist', 'artist')] = {col:encoder(le_df[col].values) for col, encoder in encoders.items()}
                    
                    # -------------------------USER->TRACKS-------------------------
                    le_df = preprocess_listen_events_df(df, type='track')
                    graph_data[('user', 'listened_to_track', 'track')]=(th.tensor(le_df['user_id'].values), th.tensor(le_df['track_id'].values))
                    edge_data_features[('user', 'listened_to_track', 'track')] = {col:encoder(le_df[col].values) for col, encoder in encoders.items()}

                    del le_df
                else:
                    raise Exception('filename in processed directory is bad.. Filename:',filename)
            
                
                del df

            # create graph data
            self.graph = heterograph(graph_data)
            
            # init graph edge data
            for edge in edge_data_features:
                for feature in edge_data_features[edge].keys():
                    feature_data = edge_data_features[edge][feature]
                    self.graph.edges[edge].data[feature] = feature_data
            del edge_data_features

            # init graph node data
            for node in node_data_features:
                for feature in node_data_features[node].keys():
                    feature_data = node_data_features[node][feature]
                    self.graph.nodes[node].data[feature] = feature_data
            del node_data_features

            # add reverse edges to the graph object
            self.graph = add_reverse_edges(self.graph)

    def __getitem__(self, idx):
        glist,_=self.load()
        return glist[idx]

    def __len__(self):
        return 1
 