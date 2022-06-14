from email import header
from re import X
import subprocess
from unittest import expectedFailure
from rsa import verify
import torch as th
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from dgl.dataloading import heterograph
from dgl.utils import extract_node_subframes, set_new_frames

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

global verifyIds_count
verifyIds_count=0

class SequenceEncoder(object):
    '''Converts a list of unique string values into a PyTorch tensor`'''
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    @th.no_grad()
    def __call__(self, list):
        return self.model.encode(list, show_progress_bar=True,convert_to_tensor=True, device=self.device)

class CategoricalEncoder(object):
    '''Converts a list of string categorical values into a PyTorch tensor`'''
    def __init__(self, device=None):
        self.device = device
    def __call__(self, list):
        categories = set(category for category in list)
        mapping = {category: i for i, category in enumerate(categories)}
        x = th.zeros(len(list), len(mapping), device=self.device)
        for i, category in enumerate(list):
            x[i, mapping[category]] = 1
        return x.to(device=self.device)

class IdentityEncoder(object):
    '''Converts a list of floating-point values into a PyTorch tensor`'''
    def __init__(self, dtype=None, device=None):
        self.dtype = dtype
        self.device = device
    def __call__(self, list):
        return th.Tensor(list).view(-1, 1).to(self.dtype).to(self.device)

def add_reverse_edges(hg, copy_ndata=True, copy_edata=True):
    '''adds reverse edges with identical feartures for every edges in graph`'''
    canonical_etypes = hg.canonical_etypes
    num_nodes_dict = {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes}
    edge_dict = {}
    for etype in canonical_etypes:
        u, v = hg.edges(form='uv', order='eid', etype=etype)
        edge_dict[etype] = (u, v)
        edge_dict[(etype[2], etype[1] + '-rev', etype[0])] = (v, u)
    new_hg = heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
    if copy_ndata:
        node_frames = extract_node_subframes(hg, None)
        set_new_frames(new_hg, node_frames=node_frames)
    if copy_edata:
        for etype in canonical_etypes:
            edge_frame = hg.edges[etype].data
            for data_name, value in edge_frame.items():
                new_hg.edges[etype].data[data_name] = value
    return new_hg

def getType(col):
    '''returns dtype of a specified column name'''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played']:
        return 'int'
    else:
        return 'str'

def setType(x, col):
    '''returns a converted value corresponding to the specified column name'''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played']:
        return int(x)
    else:
        return str(x)

def isValid(x,col):
    '''returns bool if value corresponding to the specified column name is of the correct dtype'''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played']:
        try:
            x=setType(x, col)
            return True
        except:
            return False
    else:
        try:
            x=setType(x, col)
            return True
        except:
            return False

def mapIds(df, cols, mappings):
    '''returns a dictionary mapping of id values and there index'''
    for i,col in enumerate(cols):
        mapping=mappings[i]
        new_ids=[]
        for id in df[col].tolist():
            new_ids.append(mapping[id])
        df[col]=new_ids
    return df

def get_col_names(type):
    '''returns a list of the column names corresponding to the type of data being observed'''
    if type=='user':
        return ['user_id', 'country', 'age', 'gender', 'playcount', 'registered_unixtime']
    elif type=='album':
        return ['album_id', 'album_name', 'artist_id']
    elif type=='artist':
        return ['artist_id', 'artist_name']
    elif type=='track':
        return ['track_id', 'track_name', 'artist_id']
    elif type=='le':
        return ['user_id', 'artist_id', 'album_id', 'track_id', 'timestamp']
    else:
        raise Exception('bad "type" parameter in get_col_names')

def get_fileSize(path):
    '''returns a integer val of te size of any specified file'''
    proc=subprocess.Popen(f'wc -l {path}', shell=True, stdout=subprocess.PIPE)
    return int(bytes.decode(proc.communicate()[0]).split(' ')[0])

def load_txt_df(file_path, type, load_raw=False, return_unique_ids=False, id_list=None, return_with_bad_names=False, nrows=None):
    '''returns a dataframe, names in the dataframe that are "bad", and unique values that exist in the dataframe'''
    chunksize=100000 # set chunksize
    if load_raw==False or len(get_col_names(type))==6: # if there is a header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize, nrows=nrows)
    else:# no header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', chunksize=chunksize, nrows=nrows)
    # if we only want the values in certain columns
    if return_unique_ids:
        unique_ids_dict={k: list() for k in id_list}
    else: # if we only want the full dataframe
        print(f'loading df at {file_path}')
        chunks=[]
    size = get_fileSize(file_path) # get size of file
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy() # try to set type
            except:
                try:
                    chunk[col]=[setType(x,col) for x in chunk[col]].copy() # try to force type
                    chunk[col]=chunk[col].astype(getType(col)).copy()
                except: # remove bad vals and force type
                    bad_vals=[]
                    for x in chunk[col].unique().tolist():
                        if isValid(x,col) ==False:
                            bad_vals.append(x)
                    bad_vals=np.unique(bad_vals)
                    chunk=chunk[~chunk[col].isin(bad_vals)].copy()
                    chunk[col]=[setType(x,col) for x in chunk[col]]
                    chunk[col]=chunk[col].astype(getType(col)).copy()
        if return_unique_ids:
            # all the columns are of the correct datatype now
            for id_col in id_list: # for every column we want to return
                unique_ids_dict[id_col]+=list(chunk[id_col].unique().tolist()) # set unique values of the column 
        else:
            # all the columns are of the correct datatype now
            chunks.append(chunk) # append chunk in preparation for concatenation
    if return_unique_ids: 
        for k,v in unique_ids_dict.items(): 
            unique_ids_dict[k]=list(set(v)) # make final unique values of every column in case of redundant ids
        return unique_ids_dict
    else:
        df = pd.concat(chunks) # concatenate chunks 
        if return_with_bad_names == True: # if we want the ids of the rows with bad names all get bad names function
            return df, get_bad_name_ids(df, get_col_names(type)[0])
        else:
            return df
 
   
        
def preprocess_listen_events_df(df, type):
    '''returns a dataframe, names in the dataframe that are "bad", and unique values that exist in the dataframe'''
    print(f'getting playcounts for {type}s')
    if type == 'album':
        grouped_df = df.groupby(['user_id','album_id']).size().reset_index(name="playcount")
        grouped_df['last_played'] = df.groupby(['user_id','album_id'])['timestamp'].agg('max').reset_index(name="last_played")['last_played']
    elif type == 'artist':
        grouped_df = df.groupby(['user_id','artist_id']).size().reset_index(name="playcount")
        grouped_df['last_played'] = df.groupby(['user_id','artist_id'])['timestamp'].agg('max').reset_index(name="last_played")['last_played']
    elif type == 'track':
        grouped_df = df.groupby(['user_id','track_id']).size().reset_index(name="playcount")
        grouped_df['last_played'] = df.groupby(['user_id','track_id'])['timestamp'].agg('max').reset_index(name="last_played")['last_played'] 
    else:
        raise Exception('Wrong type passed to preprocess_listen_events_df()... passed were',type)
    # This may be redundant
    for col in grouped_df:
        grouped_df[col]=grouped_df[col].astype(getType(col))
    return grouped_df

def get_bad_name_ids(df, id_type):
    col_name_id = id_type.replace('_id','_name')
    df=df[df[col_name_id]=='nan']
    bad_ids=[int(row[id_type]) for i, row in df.iterrows()] 
    return bad_ids

def get_bad_ids(key_ids, ids_list, id_type):
    found_ids=[]
    for list in ids_list:
        found_ids+=[int(x) for x in list if isinstance(x,int)]
    found_ids=np.unique(found_ids)
    df=pd.DataFrame({id_type:pd.Series(found_ids)})
    result = df[~df[id_type].isin(key_ids)][id_type].tolist()
    return result

def filterLEs(input_path, type, output_path, bad_ids, cols_to_filter, load_raw=True, nrows=None):
    chunksize=10000
    column_names=get_col_names(type='le')
    if load_raw==False:
        df_chunks = pd.read_csv(input_path, names=column_names, sep="\t", encoding='utf8', header = 0, chunksize=chunksize, nrows=nrows)
    else:
        df_chunks = pd.read_csv(input_path, names=column_names, sep="\t", encoding='utf8', chunksize=chunksize, nrows=nrows)
    size = get_fileSize(input_path)
    for i, chunk in enumerate(tqdm(df_chunks, total=size//chunksize)):
        for col, ids_list in zip(cols_to_filter,bad_ids):
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy()
            except:
                try:
                    chunk[col]=[setType(x,col) for x in chunk[col]].copy()
                    chunk[col]=chunk[col].astype(getType(col)).copy()
                except:
                    bad_vals=[]
                    for x in chunk[col].unique().tolist():
                        if isValid(x,col) ==False:
                            bad_vals.append(x)
                    bad_vals=np.unique(bad_vals)
                    print('chunk bad vals',bad_vals)
                    chunk=chunk[~chunk[col].isin(bad_vals)].copy()
                    chunk[col]=[setType(x,col) for x in chunk[col]]
                    chunk[col]=chunk[col].astype(getType(col)).copy()
            chunk = chunk[~chunk[col].isin(ids_list)].copy()
        if i==0:
            chunk.to_csv(output_path, columns=get_col_names(type), sep="\t", encoding='utf8', index=False, header=True, mode='w')
        else:
            chunk.to_csv(output_path, columns=get_col_names(type), sep="\t", encoding='utf8', index=False, header=False, mode='a')

def filterRaw(type, ids, df, output_path, fix_user_entires=False, artist_ids=None):
    col_names=get_col_names(type)
    df = df[df[col_names[0]].isin(ids)].copy()
    if artist_ids!=None:
        df = df[df['artist_id'].isin(artist_ids)].copy()
    if fix_user_entires == True:
        df['country']=df['country'].replace('nan','NoCountry').copy()
        df['gender']=df['gender'].replace('nan','NoGender').copy()
    df.to_csv(output_path, columns=col_names, sep="\t", encoding='utf8', mode='w', index=False, line_terminator='\n')



def preprocess_raw(raw_path, preprocessed_path, nrows=None):
    
    if nrows != None and os.path.exists(preprocessed_path+'/LFM-1b_LEs.txt') == False:
        print(f'----------------------------                     Making Subset of Raw Files with the first {nrows} LEs                  ----------------------------')
        unique_le_ids = load_txt_df(raw_path+'/LFM-1b_LEs.txt', type='le', load_raw=True, return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
        df, bad_artist_name_ids = load_txt_df(raw_path+'/LFM-1b_artists.txt', type='artist', load_raw=True, return_with_bad_names=True)
        artist_artist_ids=df['artist_id'].unique().tolist()
        df, bad_album_name_ids = load_txt_df(raw_path+'/LFM-1b_albums.txt', type='album', load_raw=True, return_with_bad_names=True)
        album_album_ids=df['album_id'].unique().tolist()
        album_artist_ids=df['artist_id'].unique().tolist()
        df, bad_track_name_ids = load_txt_df(raw_path+'/LFM-1b_tracks.txt', type='track', load_raw=True, return_with_bad_names=True)
        track_track_ids=df['track_id'].unique().tolist()
        track_artist_ids=df['artist_id'].unique().tolist()
        del df
        print('----------------------------                     Getting Bad Artist Ids                      ----------------------------')
        total_bad_artist_ids = np.unique(get_bad_ids(artist_artist_ids, [album_artist_ids,track_artist_ids, unique_le_ids['artist_id']], 'artist_id')+bad_artist_name_ids)
        print(f'{len(total_bad_artist_ids)} total_bad_artist_ids')
        print('----------------------------                     Getting Bad Album Ids                       ----------------------------')
        total_bad_album_ids = np.unique(get_bad_ids(album_album_ids, [unique_le_ids['album_id']], id_type='album_id')+bad_album_name_ids)
        print(f'{len(total_bad_album_ids)} total_bad_album_ids')
        print('----------------------------                     Getting Bad Track Ids                       ----------------------------')
        total_bad_track_ids = np.unique(get_bad_ids(track_track_ids, [unique_le_ids['track_id']], id_type='track_id')+bad_track_name_ids)
        print(f'{len(total_bad_track_ids)} total_bad_track_ids')
        print('---------------------------- Filtering All Bad Ids From LEs and Collecting remaining "ids"   ----------------------------')
        filterLEs(raw_path+'/LFM-1b_LEs.txt', type='le', output_path=preprocessed_path+'/LFM-1b_LEs.txt', bad_ids=[total_bad_artist_ids,total_bad_album_ids,total_bad_track_ids], cols_to_filter=['artist_id','album_id','track_id'], nrows=nrows)
        print('---------------------------- Loading Pre-Processed LEs   ----------------------------')
        unique_le_ids = load_txt_df(preprocessed_path+'/LFM-1b_LEs.txt', type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id', 'user_id'])
        print('from FilteredLes')
        for k, v in unique_le_ids.items():
            print(k, len(v))
        print('----------------------------                 Filtering Original Users File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_users.txt', type='user', load_raw=True)
        filterRaw('user', unique_le_ids['user_id'], df, preprocessed_path+'/LFM-1b_users.txt', fix_user_entires=True)
        print('----------------------------                 Filtering Original Artists File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_artists.txt', type='artist', load_raw=True)
        filterRaw('artist', unique_le_ids['artist_id'], df, preprocessed_path+'/LFM-1b_artists.txt')
        print('----------------------------                 Filtering Original Albums File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_albums.txt', type='album', load_raw=True)
        filterRaw('album', unique_le_ids['album_id'], df, preprocessed_path+'/LFM-1b_albums.txt', artist_ids=unique_le_ids['artist_id'])
        print('----------------------------                 Filtering Original Tracks File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_tracks.txt', type='track', load_raw=True)
        filterRaw('track', unique_le_ids['track_id'], df, preprocessed_path+'/LFM-1b_tracks.txt',artist_ids=unique_le_ids['artist_id'])
        del df
        print(f'subset of first {nrows} LEs made!')

        preprocess_raw_(raw_path, preprocessed_path)


    




def preprocess_raw_(raw_path, preprocessed_path):
    if os.path.exists(preprocessed_path+'/LFM-1b_LEs.txt') == False:
        print('----------------------------                     Preprocessing Raw Files                     ----------------------------')
        unique_le_ids = load_txt_df(raw_path+'/LFM-1b_LEs.txt', type='le', load_raw=True, return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
        df, bad_artist_name_ids = load_txt_df(raw_path+'/LFM-1b_artists.txt', type='artist', load_raw=True, return_with_bad_names=True)
        artist_artist_ids=df['artist_id'].unique().tolist()
        df, bad_album_name_ids = load_txt_df(raw_path+'/LFM-1b_albums.txt', type='album', load_raw=True, return_with_bad_names=True)
        album_album_ids=df['album_id'].unique().tolist()
        album_artist_ids=df['artist_id'].unique().tolist()
        df, bad_track_name_ids = load_txt_df(raw_path+'/LFM-1b_tracks.txt', type='track', load_raw=True, return_with_bad_names=True)
        track_track_ids=df['track_id'].unique().tolist()
        track_artist_ids=df['artist_id'].unique().tolist()
        del df
        print('----------------------------                     Getting Bad Artist Ids                      ----------------------------')
        total_bad_artist_ids = np.unique(get_bad_ids(artist_artist_ids, [album_artist_ids,track_artist_ids, unique_le_ids['artist_id']], 'artist_id')+bad_artist_name_ids)
        print(f'{len(total_bad_artist_ids)} total_bad_artist_ids')
        print('----------------------------                     Getting Bad Album Ids                       ----------------------------')
        total_bad_album_ids = np.unique(get_bad_ids(album_album_ids, [unique_le_ids['album_id']], id_type='album_id')+bad_album_name_ids)
        print(f'{len(total_bad_album_ids)} total_bad_album_ids')
        print('----------------------------                     Getting Bad Track Ids                       ----------------------------')
        total_bad_track_ids = np.unique(get_bad_ids(track_track_ids, [unique_le_ids['track_id']], id_type='track_id')+bad_track_name_ids)
        print(f'{len(total_bad_track_ids)} total_bad_track_ids')
        print('---------------------------- Filtering All Bad Ids From LEs and Collecting remaining "ids"   ----------------------------')
        filterLEs(raw_path+'/LFM-1b_LEs.txt', type='le', output_path=preprocessed_path+'/LFM-1b_LEs.txt', bad_ids=[total_bad_artist_ids,total_bad_album_ids,total_bad_track_ids], cols_to_filter=['artist_id','album_id','track_id'])

    condition=os.path.exists(preprocessed_path+'/LFM-1b_users.txt') == False or os.path.exists(preprocessed_path+'/LFM-1b_artists.txt') == False or os.path.exists(preprocessed_path+'/LFM-1b_albums.txt') == False or os.path.exists(preprocessed_path+'/LFM-1b_tracks.txt') == False
    if condition:
        print('---------------------------- Loading Pre-Processed LEs   ----------------------------')
        unique_le_ids = load_txt_df(preprocessed_path+'/LFM-1b_LEs.txt', type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id', 'user_id'])
        print('from FilteredLes')
        for k, v in unique_le_ids.items():
            print(k, len(v))
        print('----------------------------                 Filtering Original Users File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_users.txt', type='user', load_raw=True)
        filterRaw('user', unique_le_ids['user_id'], df, preprocessed_path+'/LFM-1b_users.txt', fix_user_entires=True)
        print('----------------------------                 Filtering Original Artists File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_artists.txt', type='artist', load_raw=True)
        filterRaw('artist', unique_le_ids['artist_id'], df, preprocessed_path+'/LFM-1b_artists.txt')
        print('----------------------------                 Filtering Original Albums File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_albums.txt', type='album', load_raw=True)
        filterRaw('album', unique_le_ids['album_id'], df, preprocessed_path+'/LFM-1b_albums.txt', artist_ids=unique_le_ids['artist_id'])
        print('----------------------------                 Filtering Original Tracks File                    ----------------------------')
        df = load_txt_df(raw_path+'/LFM-1b_tracks.txt', type='track', load_raw=True)
        filterRaw('track', unique_le_ids['track_id'], df, preprocessed_path+'/LFM-1b_tracks.txt',artist_ids=unique_le_ids['artist_id'])
        del df