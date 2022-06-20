import os
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from data_utils import get_fileSize, get_raw_df, get_raw_ids, get_preprocessed_ids, get_bad_name_ids, get_bad_ids, get_col_names, getType, isValid, setType

def filterRaw(type, ids, df_path, output_path, fix_user_entires=False, artist_ids=None):
    print(f'----------------------------                 Filtering Original {type} File                    ----------------------------')
    col_names=get_col_names(type)
    df = get_raw_df(df_path, type=type)
    df = df[df[col_names[0]].isin(ids)].copy()
    if artist_ids!=None:
        df = df[df['artist_id'].isin(artist_ids)].copy()
    if fix_user_entires == True:
        df['country']=df['country'].replace('nan','NoCountry').copy()
        df['gender']=df['gender'].replace('nan','NoGender').copy()
    df.to_csv(output_path, columns=col_names, sep="\t", encoding='utf8', mode='w', index=False, line_terminator='\n')

def filterLEs(input_path, type, output_path, bad_ids, cols_to_filter, load_raw=True, nrows=None):
    ''' 
    the filterLEs reads the LEs from a specified input file and filters the ids based on each 
    list in bad_ids specified by the id column in cols_to_filter. After filtered the data is saved in the output path
    '''
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




def preprocess_raw(raw_path, preprocessed_path, nrows=None, overwrite=False):
    """
    The preprocess_raw function is the brains of the data manipulation required to form a fully connected graph of the LFM-1b dataset
    With the raw directory, preprocessed directory, number of rows to sample, and an overwrite inidcator. 

    This function makes a subset if and only if the nrows parameter is not a None. If not making a subset, this function reads the raw
    file directory and stripts the listen events file of all ids of artists/tracks/albums/users whose name is unparsable, or whose id doesn't 
    exist inside the respective artist/album/track/user file.

    Once cleaned the listen events file is saved and the artist/album/track/user files are updated based on the existing unique ids 
    in the cleaned listen events file
    """
    
    condition= os.path.exists(preprocessed_path+'/LFM-1b_LEs.txt') == False
    if condition == True or nrows!=None:
        les_raw_path=raw_path+'/LFM-1b_LEs.txt'
        artists_raw_path=raw_path+'/LFM-1b_artists.txt'
        albums_raw_path=raw_path+'/LFM-1b_albums.txt'
        tracks_raw_path=raw_path+'/LFM-1b_tracks.txt'
        users_raw_path=raw_path+'/LFM-1b_users.txt'

        les_pre_path=preprocessed_path+'/LFM-1b_LEs.txt'
        artists_pre_path=preprocessed_path+'/LFM-1b_artists.txt'
        albums_pre_path=preprocessed_path+'/LFM-1b_albums.txt'
        tracks_pre_path=preprocessed_path+'/LFM-1b_tracks.txt'
        users_pre_path=preprocessed_path+'/LFM-1b_users.txt'
        if overwrite==True:
            if nrows:
                print(f'----------------------------                     Preprocessing Subset size {nrows} from Raw Files                 ----------------------------')
            else:
                print('----------------------------                     Preprocessing Raw Files                 ----------------------------')
            

            unique_le_ids = get_raw_ids(les_raw_path, type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
            artist_id_dict = get_raw_ids(artists_raw_path, type='artist', return_unique_ids=True, id_list=['artist_id'])
            bad_artist_name_ids = get_bad_name_ids(artists_raw_path, type='artist')
            album_id_dict=get_raw_ids(albums_raw_path, type='album', return_unique_ids=True, id_list=['album_id','artist_id'])
            bad_album_name_ids = get_bad_name_ids(albums_raw_path, type='album')
            track_id_dict=get_raw_ids(tracks_raw_path, type='track', return_unique_ids=True, id_list=['track_id','artist_id'])
            bad_track_name_ids = get_bad_name_ids(tracks_raw_path, type='track')
            
            print('---------------------------- Filtering All Bad Ids From LEs and Collecting remaining "ids"   ----------------------------')
            total_bad_artist_ids = np.unique(get_bad_ids(artist_id_dict['artist_id'], [album_id_dict['artist_id'],track_id_dict['artist_id'], unique_le_ids['artist_id']], 'artist_id')+bad_artist_name_ids)
            total_bad_album_ids = np.unique(get_bad_ids(album_id_dict['album_id'], [unique_le_ids['album_id']], id_type='album_id')+bad_album_name_ids)
            total_bad_track_ids = np.unique(get_bad_ids(track_id_dict['track_id'], [unique_le_ids['track_id']], id_type='track_id')+bad_track_name_ids)
            filterLEs(les_raw_path, type='le', output_path=les_pre_path, bad_ids=[total_bad_artist_ids,total_bad_album_ids,total_bad_track_ids], cols_to_filter=['artist_id','album_id','track_id'], nrows=nrows)

        if os.path.exists(preprocessed_path+'/LFM-1b_users.txt') == False or os.path.exists(preprocessed_path+'/LFM-1b_artists.txt') == False or os.path.exists(preprocessed_path+'/LFM-1b_albums.txt') == False or os.path.exists(preprocessed_path+'/LFM-1b_tracks.txt') == False:
            unique_le_ids = get_preprocessed_ids(les_pre_path, type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
            
            filterRaw('user', unique_le_ids['user_id'], users_raw_path, users_pre_path, fix_user_entires=True)
            filterRaw('artist', unique_le_ids['artist_id'], artists_raw_path, artists_pre_path)
            filterRaw('album', unique_le_ids['album_id'], albums_raw_path, albums_pre_path, artist_ids=unique_le_ids['artist_id'])
            filterRaw('track', unique_le_ids['track_id'], tracks_raw_path, tracks_pre_path, artist_ids=unique_le_ids['artist_id'])
        
            print('---------------------------- Loading Pre-Processed LEs   ----------------------------')
            file_path=raw_path+'_UGP/genres_allmusic.txt'
            df = pd.read_csv(file_path, names=['genre_name'])
            df['genre_id']=df['genre_name'].index
            df=df.reindex(columns=['genre_id', 'genre_name'])
            output_path=preprocessed_path+'/genres_allmusic.txt'
            df.to_csv(output_path, columns=get_col_names('genre'), sep="\t", encoding='utf8', mode='w', index=False, line_terminator='\n')

            del df
        