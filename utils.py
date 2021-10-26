from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle

def name_particular_rooms(path,rooms):
    if rooms == None:
        print('rooms == None')
        return
    suffixes = '_'.join(['{}-{}'.format(k,v) for k ,v in rooms.items()])
    file = os.path.join(path,'../','names','names_{}.pkl'.format(suffixes))
    if os.path.exists(file):
        print('names_{}.pkl exists'.format(suffixes))
        return
    names = os.listdir(path)
    names_ = []
    for name in tqdm(names):
        with open(os.path.join(path,name),'rb') as pkl_file:
            layout = pickle.load(pkl_file)
        
        if np.prod([len(layout[k])==v for k,v in rooms.items()])>0:
            names_.append(name)
    with open(file, 'wb') as output:
        pickle.dump(names_, output)
    print('find {} layouts satisfying the rooms requirement'.format(len(names_)))
    return 

def types_more_than_n(path,n=2000):
    file = os.path.join(path,'../','names','morethan_{}.pkl'.format(str(n)))
    if os.path.exists(file):
        print('morethan_{}.pkl exists'.format(str(n)))
        return
    names = os.listdir(path)
    rooms=range(10)#(0,1)
    num_types={}
    for name in tqdm(names):
        with open(os.path.join(path,name),'rb') as pkl_file:
            layout = pickle.load(pkl_file)
        if '_'.join([str(len(layout[r])) for r in rooms]) in num_types.keys():
            num_types['_'.join([str(len(layout[r])) for r in rooms])]+=1
        else:
            num_types['_'.join([str(len(layout[r])) for r in rooms])]=1
    num_types = pd.DataFrame.from_dict(num_types,orient='index')
    num_types = num_types.sort_values(0,ascending=True)
    types = list(num_types[num_types>2000].dropna().index)

    names_ = []
    for name in tqdm(names):
        with open(os.path.join(path,name),'rb') as pkl_file:
            layout = pickle.load(pkl_file)
        if '_'.join([str(len(layout[r])) for r in rooms]) in types:
            names_.append(name)
    with open(file, 'wb') as output:
        pickle.dump(names_, output)
    print('find {} types - {} layouts out of {} in total'.format(len(types),len(names_),len(names)))
    return 

