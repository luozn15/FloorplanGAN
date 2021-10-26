import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from scipy.stats import norm
#from sklearn.preprocessing import  OneHotEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import pickle
import load_data

def generate_random_layout(dataset, batch_size):
    rooms = dataset.rect_types[np.random.randint(dataset.rect_types.shape[0], size=batch_size), :]
    mask = rooms<0
    mask_tile = np.tile(np.expand_dims(mask,2),dataset.enc_len+4)
    rooms[rooms<0] = 0
    #房间数量
    #num_rects = np.around(np.random.normal(loc=dataset.mu_num,scale=dataset.sigma_num,size=(batch_size,)))
    #num_rects = np.maximum(num_rects,1)
    #num_rects = np.minimum(num_rects,dataset.maximum_elements_num)

    #mask = (np.arange(dataset.maximum_elements_num) < num_rects.reshape(-1,1)).astype(np.uint8)
    

    #N,46,9
    
    #rooms=np.random.choice(dataset.rect_types,size=(batch_size,dataset.maximum_elements_num),replace=True)

    #rooms = np.sort(rooms)
    #encoded = dataset.enc.transform(rooms.reshape(-1,1)).toarray().reshape(batch_size,dataset.maximum_elements_num,-1)
    encoded = F.one_hot(torch.tensor(np.vectorize(lambda x:dataset.index_mapping[x])(rooms)).to(torch.int64),\
                        num_classes=dataset.enc_len).numpy().astype(np.float32)
    #N,46,4
    loc = np.array([np.random.multivariate_normal(mean=dataset.mean[room], cov=dataset.cov[room]) for room in rooms.flatten()])\
        .reshape((rooms.shape[0],rooms.shape[1],4))/dataset.img_size
    loc[loc<0]=0
    loc[loc>1]=1
    x0,y0,x1,y1 = loc[:,:,0], loc[:,:,1], loc[:,:,2], loc[:,:,3]
    xc = np.expand_dims((x0+x1)/2,-1)
    w = np.expand_dims(abs(x1-x0),-1)
    w[w<=0]=0.02
    yc = np.expand_dims((y0+y1)/2,-1)
    h = np.expand_dims(abs(y1-y0),-1)
    h[h<=0]=0.02
    #area_root = (w*h)**0.5

    #N,46,13
    v = np.concatenate([encoded, xc, yc, w, h],axis=2)
    '''order_roomtype = np.argsort(rooms,axis=-1)

    sorted_by_roomtype = (v*mask)[np.arange(batch_size)[:, np.newaxis],order_roomtype]
    order_zeros = np.argsort(-sorted_by_roomtype[:,:,:-4].sum(axis=-1),axis=-1)
    sorted_by_zeros=sorted_by_roomtype[np.arange(batch_size)[:, np.newaxis],order_zeros]
    mask_padding = (np.arange(dataset.maximum_elements_num) >= num_rects.reshape(-1,1)).astype(np.bool)
    return sorted_by_zeros.astype(np.float32), mask_padding'''
    return (v*(~mask_tile)).astype(np.float32), mask


class wireframeDataset_Rplan(Dataset):
    def __init__(self, path='../../data_RPLAN/floorplan_dataset/pkls',subset_name=False, img_size=256): #rooms={0:1,1:1}
        self.path = path
        #self.maximum_elements_num = maximum_elements_num
        self.dict_room_encode ={
            'Living room':0,
            'Master room':1,
            'Kitchen':2,
            'Bathroom':3,
            'Dining room':4,
            'Child room':5,
            'Study room':6,
            'Second room':7,
            'Guest room':8,
            'Balcony':9
                    }

        if subset_name:
            file = os.path.join(path,'..','names',subset_name)
            with open(file, 'rb') as pkl_file:
                self.names = pickle.load(pkl_file)
        else:
            self.names = os.listdir(path) 

        temp = []
        for name in tqdm(self.names):#统计房间类型
            with open(os.path.join(self.path,name),'rb') as pkl_file:
                floorplan = pickle.load(pkl_file)
            for k,v in floorplan.items():
                if len(v)>0 and k not in temp:
                    temp.append(k)
        self.rooms = {v:k for k,v in self.dict_room_encode.items() if v in temp}
        #self.rooms_reindex = {i:v for i,v in enumerate(self.rooms.values())}
        self.index_mapping = {v:i for i,v in enumerate(self.rooms.keys())}

        #self.enc = OneHotEncoder()
        #self.enc.fit(np.array(range(len(self.dict_room_encode))).reshape(-1,1))
        #self.enc.fit(np.array(list(self.dict_room_encode.values())).reshape(-1,1))
        self.enc_len =len(self.rooms)
        self.img_size = img_size

        _ = self.get_statistics()
        
    def __getitem__(self,index):
        name = self.names[index]
        with open(os.path.join(self.path,name),'rb') as pkl_file:
            layout = pickle.load(pkl_file)
        layout = {key:layout[key] for key in self.rooms.keys()}
        data = []
        for room,_list in layout.items():
            #ans =self.enc.transform(np.array([room]).reshape(-1,1)).toarray()
            ans = F.one_hot(torch.tensor(self.index_mapping[room]), num_classes=self.enc_len).numpy()
            for r in _list:
                (x0,y1),(x1,y0) = r
                x0,y1,x1,y0 = x0/self.img_size, y1/self.img_size, x1/self.img_size, y0/self.img_size
                xc = (x0+x1)/2
                w = abs(x1-x0)
                yc = (y0+y1)/2
                h = abs(y1-y0)
                #area_root = (w*h)**0.5

                v = [ans,np.array([xc,yc,w,h])]
                #v = [ans,np.array([x0,y0,x1,y1])]
                v = np.concatenate(v)
                data.append(v)
        data = np.array(data).reshape(-1,self.enc_len + 4)
        '''if data.shape[0]>=self.maximum_elements_num:
            data_ = data[:self.maximum_elements_num].astype(np.float32)
            return data_, self.maximum_elements_num
        else:'''
        num_elements_to_fill = self.maximum_elements_num-data.shape[0]
        to_fill = np.zeros((num_elements_to_fill,data.shape[1]))
        data_ = np.concatenate([data,to_fill],axis=0).astype(np.float32)
        mask = np.concatenate([np.zeros(data.shape[0]),np.ones(num_elements_to_fill)]).astype(np.bool)
        return data_, mask

    def __len__(self):
        return len(self.names)

    def get_statistics(self):
        if hasattr(self,'mu'):
            return {'mu_roomnumber':self.mu_num, 
            'sigma_roomnumber':self.sigma_num,  
            'mean_loc':self.mean, 
            'cov_loc':self.cov,
            'rect_types':self.rect_types,
            #'X':self.X,
            'maximum_elements_num':self.maximum_elements_num }

        num_rects={}#房间数量
        rect_types = []#房间类型
        coordinates = {name:[] for name in self.rooms.keys()}#统计坐标(x1,y1,x2,y2)
        elements_num=[]#元素数量
        for name in tqdm(self.names):
            with open(os.path.join(self.path,name),'rb') as pkl_file:
                floorplan = pickle.load(pkl_file)
            floorplan = {key:floorplan[key] for key in self.rooms}
            num_rect=0
            element_num=0
            rect_type=[]
            for name_room,rects in floorplan.items():
                num_rect+= len(rects)
                element_num+=len(rects)
                for rect in rects:
                    rect_type.append(name_room)
                    coordinates[name_room].append(np.array(rect).reshape(-1))
            rect_types.append(rect_type)
            num_rects[name]=num_rect
            elements_num.append(element_num)
        x = np.array(list(num_rects.values()))
        self.mu_num = np.mean(x) 
        self.sigma_num = np.std(x)

        coordinates = {k:np.array(v) for k,v in coordinates.items()}
        self.mean = {k:v.mean(axis=0) for k,v in coordinates.items()}
        self.cov = {k:np.cov(v.T) for k,v in coordinates.items()}
        self.maximum_elements_num = max(elements_num)

        self.rect_types = np.array([np.pad(np.array(r),(0,self.maximum_elements_num-len(r)),'constant',constant_values=(-1,-1)) for r in rect_types],dtype=np.int)
        #长宽比统计
        #ratio_as = np.array([(r[0][:,:-4].sum(axis=-1)>0.5) * (r[0][:,-2]/r[0][:,-1]) for r in self])
        #self.ratio_as = (np.percentile(np.array(ratio), 10,axis = 0), np.percentile(np.array(ratio), 90,axis = 0))

        return {'mu_roomnumber':self.mu_num, 
            'sigma_roomnumber':self.sigma_num,  
            'mean_loc':self.mean, 
            'cov_loc':self.cov,
            'rect_types':self.rect_types,
            #'X':self.X,
            'maximum_elements_num':self.maximum_elements_num }
     
if __name__=='__main__':
    name_particular_rooms(path='../../data_RPLAN/floorplan_dataset/pkls',
                        rooms={0:1,1:1,2:1,3:1,7:1,9:1})