import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

def getType(col):
    '''returns dtype of a specified column name'''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played', 'genre_id']:
        return 'int'
    else:
        return 'str'

def setType(x, col):
    '''returns a converted value corresponding to the specified column name'''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played', 'genre_id']:
        return int(x)
    else:
        return str(x)

def isValid(x,col):
    '''returns bool if value corresponding to the specified column name is of the correct dtype'''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played', 'genre_id']:
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
    elif type=='genre':
        return ['genre_id', 'genre_name']
    else:
        raise Exception('bad "type" parameter in get_col_names')


def get_fileSize(path):
    '''returns a integer val of te size of any specified file'''
    proc=subprocess.Popen(f'wc -l {path}', shell=True, stdout=subprocess.PIPE)
    return int(bytes.decode(proc.communicate()[0]).split(' ')[0])

def get_raw_df(file_path, type):
    chunksize=100000 # set chunksize
    if len(get_col_names(type))==6: # if there is a header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    else:# no header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', chunksize=chunksize)

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
        # all the columns are of the correct datatype now
        chunks.append(chunk) # append chunk in preparation for concatenation
    df = pd.concat(chunks) # concatenate chunks 
    return df

def get_bad_ids(key_ids, ids_list, id_type):
    '''returns a list of ids of type id_type whose id is not in the id_type file, but is in the list of all available ids'''
    found_ids=[]
    for list in ids_list:
        found_ids+=[int(x) for x in list if isinstance(x,int)]
    found_ids=np.unique(found_ids)
    df=pd.DataFrame({id_type:pd.Series(found_ids)})
    result = df[~df[id_type].isin(key_ids)][id_type].tolist()
    return result 

def get_bad_name_ids(file_path, type):
    '''returns a list of ids of type id_type whose name is unparsable'''
    col_name_id = type+'_name'
    chunksize=100000 # set chunksize 
    if len(get_col_names(type))==6: # if there is a header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    else:# no header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', chunksize=chunksize)
    # if we only want the values in certain columns
    bad_ids=[] 
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
        chunk=chunk[chunk[col_name_id]=='nan']
        bad_ids+=list(chunk[type+'_id'].values) # set unique values of the column 
    return list(set(bad_ids))

def get_raw_ids(file_path, type, return_unique_ids=False, id_list=None):
    print(f'---------------------------- Loading Raw {type} file  ----------------------------')
    chunksize=100000 # set chunksize 
    if len(get_col_names(type))==6: # if there is a header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    else:# no header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', chunksize=chunksize)
    # if we only want the values in certain columns
    ids_dict={k: list() for k in id_list}
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
                ids_dict[id_col]+=list(chunk[id_col].unique().tolist()) # set unique values of the column 
        else:
            for id_col in id_list: # for every column we want to return
                ids_dict[id_col]+=list(chunk[id_col].values) # set unique values of the column 
    if return_unique_ids: 
        for k,v in ids_dict.items(): 
            ids_dict[k]=list(set(v)) # make final unique values of every column in case of redundant ids
        return ids_dict
    else:
        return ids_dict

def get_preprocessed_ids(file_path, type, return_unique_ids=False, id_list=None):
    print(f'---------------------------- Loading Preprocessed {type} file  ----------------------------')
    chunksize=100000 # set chunksize
    df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    # if we only want the values in certain columns
    ids_dict={k: list() for k in id_list}
    size = get_fileSize(file_path) # get size of file
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy() # try to set type
            except:
                raise Exception('type not correct in preprocessed file')
        if return_unique_ids:
            # all the columns are of the correct datatype now
            for id_col in id_list: # for every column we want to return
                ids_dict[id_col]+=list(chunk[id_col].unique().tolist()) # set unique values of the column 
        else:
            for id_col in id_list: # for every column we want to return
                ids_dict[id_col]+=list(chunk[id_col].values) # set unique values of the column 
    if return_unique_ids: 
        for k,v in ids_dict.items(): 
            ids_dict[k]=list(set(v)) # make final unique values of every column in case of redundant ids
        return ids_dict
    else:
        return ids_dict

global verifyIds_count
verifyIds_count=0

def get_le_playcount(file_path,type_id, user_mapping, groupby_mapping, relative_playcount=False):
    print(f'loading {type_id} playcount')
    chunksize=10000000
    df_chunks = pd.read_csv(file_path, names=get_col_names('le'), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    size = get_fileSize(file_path) # get size of file
    playcount_dict={}
    total_user_plays={}
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            chunk[col]=chunk[col].astype(getType(col)).copy() 
        for user_id, groupby_id in zip(chunk['user_id'].values, chunk[type_id].values): 
            try:
                user_id=user_mapping[user_id]
                groupby_id=groupby_mapping[groupby_id]
                if (user_id,groupby_id) not in playcount_dict.keys():
                    playcount_dict[(user_id,groupby_id)]=1
                    total_user_plays[user_id]=1
                else:
                    playcount_dict[(user_id,groupby_id)]+=1
                    total_user_plays[user_id]+=1
            except:
                pass    
    if relative_playcount:
        return [user_id for user_id, groupby_id in playcount_dict.keys()], [groupby_id for user_id, groupby_id in playcount_dict.keys()], [val/total_user_plays[u_id] for (u_id,g_id), val in playcount_dict.items()]
    else:
        return [user_id for user_id, groupby_id in playcount_dict.keys()], [groupby_id for user_id, groupby_id in playcount_dict.keys()], [val for (u_id,g_id), val in playcount_dict.items()]


def get_full_le_plays(file_path,type_id, user_mapping, groupby_mapping):
    print(f'loading {type_id} plays')
    chunksize=10000000
    df_chunks = pd.read_csv(file_path, names=get_col_names('le'), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    size = get_fileSize(file_path) # get size of file
    user_ids=[]
    groupby_ids=[]
    timestamps=[]
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            chunk[col]=chunk[col].astype(getType(col)).copy() 
        for user_id, groupby_id, timestamp in zip(chunk['user_id'].values, chunk[type_id].values, chunk['timestamp'].values): 
            try:
                user_id=user_mapping[user_id]
                groupby_id=groupby_mapping[groupby_id]

                user_ids.append(user_id)
                groupby_ids.append(groupby_id)
                timestamps.append(int(timestamp))
            except:
                pass    
    return user_ids, groupby_ids, timestamps

def remap_ids(col_dict, ordered_cols, mappings, is_pdDataframe=False):
    new_mapping={col_name:list() for col_name in col_dict.keys()}
    print('remapping ids')
    length=len(col_dict[ordered_cols[0]])
    for row_index in tqdm(range(length), total=length):
        bad_row=False
        for mapping, col_name in zip(mappings,ordered_cols):
            try:
                val=col_dict[col_name][row_index]
                new_mapping[col_name].append(mapping[val])
            except:
                bad_row=True
                print('found bad id while mapping')
        if bad_row ==False:
            for col_name in col_dict.keys():
                if col_name not in ordered_cols:
                    val=col_dict[col_name][row_index]
                    new_mapping[col_name].append(val)
    return new_mapping


def get_artist_genre_df(artist_genres_allmusic_path, artist_name_to_id_mapping):
    file=open(artist_genres_allmusic_path, 'r')
    lines=file.readlines()
    data={'artist_id':list(),'genre_id':list()}
    for line in lines:
        info=line.strip().split('\t')
        name=str(info[0])
        genre_list=np.array([int(x) for x in info[1:]])
        if len(genre_list) != 0 and name in artist_name_to_id_mapping.keys():
            for genre in genre_list:
                data['artist_id'].append(artist_name_to_id_mapping[name])
                data['genre_id'].append(genre)

    return pd.DataFrame(data)