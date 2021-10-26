import pickle
import os
import numpy as np
from PIL import Image

path ='./'

def get_pixels(path_dir):
    images = {}
    for file in os.listdir(path_dir):
        if os.path.isdir(os.path.join(path_dir,file)): continue
        image = Image.open(os.path.join(path_dir,file))
        name = file.split('.')[0]
        images[name] =  np.asanyarray(image,dtype=np.int64)
    return images

def get_plines(path_dir):
    #os.chdir(path_dir)
    plines = {}
    for file in os.listdir(path_dir):
        #print(file)
        pkl_file = open(os.path.join(path_dir,file),'rb')
        name = file.split('.')[0]
        plines[name] = pickle.load(pkl_file)
    return plines

def get_rectangle(path_dir):
    plines = get_plines(path_dir)
    rect={}
    for file, _dict in plines.items():
        new_dict={}
        for room, pls in _dict.items():
            rects=[]
            for pl in pls:
                left_top = (pl[:,0].min(),pl[:,1].max())
                right_bottom = (pl[:,0].max(),pl[:,1].min())
                rects.append((left_top,right_bottom))
            new_dict[room] = rects
        rect[file]=new_dict
    return rect


if __name__ == "__main__":
    path_dir = os.path.join(path,'dataset/'+ 'test_label' +'/pkls')
    plines = get_plines(path_dir)