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


def name_particular_rooms(path, rooms):
    if rooms == None:
        print('rooms == None')
        return
    suffixes = '_'.join(['{}-{}'.format(k, v) for k, v in rooms.items()])
    file = os.path.join(path, '..', 'names', 'names_{}.pkl'.format(suffixes))
    if os.path.exists(file):
        print('names_{}.pkl exists'.format(suffixes))
        return
    names = os.listdir(path)
    names_ = []
    maximum = [0 for r in rooms]
    for name in tqdm(names):
        with open(os.path.join(path, name), 'rb') as pkl_file:
            layout = pickle.load(pkl_file)

        if np.prod([len(layout[k]) == v for k, v in rooms.items()]) > 0:
            for i in range(len(rooms)):
                maximum[i] = max([maximum[i], len(layout[i])])
            names_.append(name)
    with open(file, 'wb') as output:
        pickle.dump(names_, output)
    print('find {} layouts satisfying the rooms requirement'.format(len(names_)))
    return


def generate_random_layout(dataset, batch_size):
    rooms = dataset.rect_types[np.random.randint(
        dataset.rect_types.shape[0], size=batch_size), :]
    mask = rooms < 0
    mask_tile = np.tile(np.expand_dims(mask, 2), dataset.enc_len+4)
    rooms[rooms < 0] = 0
    encoded = F.one_hot(torch.tensor(
        np.vectorize(lambda x: dataset.index_mapping[x])(rooms)
    ).to(torch.int64),
        num_classes=dataset.enc_len
    ).numpy().astype(np.float32)
    # N,46,4
    loc = np.array([np.random.multivariate_normal(mean=dataset.mean[room], cov=dataset.cov[room]) for room in rooms.flatten()])\
        .reshape((rooms.shape[0], rooms.shape[1], 4))/dataset.img_size
    loc[loc < 0] = 0
    loc[loc > 1] = 1
    x0, y0, x1, y1 = loc[:, :, 0], loc[:, :, 1], loc[:, :, 2], loc[:, :, 3]
    xc = np.expand_dims((x0+x1)/2, -1)
    w = np.expand_dims(abs(x1-x0), -1)
    w[w <= 0] = 0.02
    yc = np.expand_dims((y0+y1)/2, -1)
    h = np.expand_dims(abs(y1-y0), -1)
    h[h <= 0] = 0.02
    area_root = (w*h)**0.5

    # N,46,13
    v = np.concatenate([encoded, xc, yc, area_root, w], axis=2)
    '''
    order_roomtype = np.argsort(rooms,axis=-1)

    sorted_by_roomtype = (v*mask)[np.arange(batch_size)[:, np.newaxis],order_roomtype]
    order_zeros = np.argsort(-sorted_by_roomtype[:,:,:-4].sum(axis=-1),axis=-1)
    sorted_by_zeros=sorted_by_roomtype[np.arange(batch_size)[:, np.newaxis],order_zeros]
    mask_padding = (np.arange(dataset.maximum_elements_num) >= num_rects.reshape(-1,1)).astype(np.bool)
    return sorted_by_zeros.astype(np.float32), mask_padding'''
    return (v*(~mask_tile)).astype(np.float32), mask


class wireframeDataset_Rplan(Dataset):
    # rooms={0:1,1:1}
    def __init__(self, cfg, img_size=256):
        self.path = cfg.MANUAL.RPLAN_PATH
        #self.maximum_elements_num = maximum_elements_num
        self.dict_class_id = {
            'Living room': 0,
            'Master room': 1,
            'Kitchen': 2,
            'Bathroom': 3,
            'Dining room': 4,
            'Child room': 5,
            'Study room': 6,
            'Second room': 7,
            'Guest room': 8,
            'Balcony': 9
        }
        subset_name = cfg.DATASET.SUBSET
        if subset_name:
            file = os.path.join(self.path, '..', 'names', subset_name)
            with open(file, 'rb') as pkl_file:
                self.names = pickle.load(pkl_file)
        else:
            self.names = os.listdir(self.path)

        self.data = dict()  # 数据加载到内存， 以字典保存
        available_classid = set()
        # 统计房间类型
        for name in tqdm(self.names):
            with open(os.path.join(self.path, name), 'rb') as pkl_file:
                layout = pickle.load(pkl_file)
            self.data[name] = layout
            for classid, v in layout.items():
                if len(v) > 0:
                    available_classid.add(classid)
        self.dict_id_class = {classid: classname for classname, classid in self.dict_class_id.items()
                              if classid in available_classid}
        #self.rooms_reindex = {i:v for i,v in enumerate(self.rooms.values())}
        self.index_mapping = {classid: id_mapped for id_mapped,
                              classid in enumerate(self.dict_id_class.keys())}

        #self.enc = OneHotEncoder()
        # self.enc.fit(np.array(range(len(self.dict_room_encode))).reshape(-1,1))
        # self.enc.fit(np.array(list(self.dict_room_encode.values())).reshape(-1,1))
        self.enc_len = len(self.index_mapping)
        self.img_size = img_size

        _ = self.get_statistics()

    def __getitem__(self, index):
        name = self.names[index]
        layout = self.data[name]
        layout = {classid: layout[classid]
                  for classid in self.dict_id_class.keys()}
        data = []
        for classid, rects in layout.items():
            #ans =self.enc.transform(np.array([room]).reshape(-1,1)).toarray()
            onehot_enc = F.one_hot(torch.tensor(
                self.index_mapping[classid]), num_classes=self.enc_len).numpy()
            for r in rects:
                (x0, y1), (x1, y0) = r
                x0, y1, x1, y0 = x0/self.img_size, y1 / \
                    self.img_size, x1/self.img_size, y0/self.img_size
                xc = (x0+x1)/2
                w = abs(x1-x0)
                yc = (y0+y1)/2
                h = abs(y1-y0)
                area_root = (w*h)**0.5

                v = [onehot_enc, np.array([xc, yc, area_root, w])]
                #v = [ans,np.array([x0,y0,x1,y1])]
                v = np.concatenate(v)
                data.append(v)
        data = np.array(data).reshape(-1, self.enc_len + 4)
        '''if data.shape[0]>=self.maximum_elements_num:
            data_ = data[:self.maximum_elements_num].astype(np.float32)
            return data_, self.maximum_elements_num
        else:'''
        num_elements_to_fill = self.maximum_elements_num-data.shape[0]
        to_fill = np.zeros((num_elements_to_fill, data.shape[1]))
        data_ = np.concatenate([data, to_fill], axis=0).astype(np.float32)
        mask = np.concatenate([np.zeros(data.shape[0]), np.ones(
            num_elements_to_fill)]).astype(np.bool)
        return data_, mask

    def __len__(self):
        return len(self.names)

    def get_statistics(self):
        if hasattr(self, 'mu'):
            return {'mu_roomnumber': self.mu_num,
                    'sigma_roomnumber': self.sigma_num,
                    'mean_loc': self.mean,
                    'cov_loc': self.cov,
                    'rect_types': self.rect_types,
                    # 'X':self.X,
                    'maximum_elements_num': self.maximum_elements_num}

        dict_samplename_Nrects = {}  # 样本名: 房间数量
        rect_types = []  # 房间类型
        coordinates = {classid: []
                       for classid in self.dict_id_class.keys()}  # 统计坐标(x1,y1,x2,y2)
        for name, layout in tqdm(self.data.items()):
            layout = {classid: layout[classid]
                      for classid in self.dict_id_class}
            num_rect = 0
            rect_type = []
            for classid, rects in layout.items():
                num_rect += len(rects)
                for rect in rects:
                    rect_type.append(classid)
                    coordinates[classid].append(np.array(rect).reshape(-1))
            rect_types.append(rect_type)
            dict_samplename_Nrects[name] = num_rect
        x = np.array(list(dict_samplename_Nrects.values()))
        self.mu_num = np.mean(x)
        self.sigma_num = np.std(x)

        coordinates = {classid: np.array(v)
                       for classid, v in coordinates.items()}
        self.mean = {classid: v.mean(axis=0)
                     for classid, v in coordinates.items()}
        self.cov = {classid: np.cov(v.T) for classid, v in coordinates.items()}
        self.maximum_elements_num = max(list(dict_samplename_Nrects.values()))

        self.rect_types = np.array(
            [np.pad(
                np.array(r),
                (0, self.maximum_elements_num-len(r)),
                'constant',
                constant_values=(-1, -1)
            )
                for r in rect_types],
            dtype=np.int
        )
        # 长宽比统计
        #ratio_as = np.array([(r[0][:,:-4].sum(axis=-1)>0.5) * (r[0][:,-2]/r[0][:,-1]) for r in self])
        #self.ratio_as = (np.percentile(np.array(ratio), 10,axis = 0), np.percentile(np.array(ratio), 90,axis = 0))

        return {'mu_roomnumber': self.mu_num,
                'sigma_roomnumber': self.sigma_num,
                'mean_loc': self.mean,
                'cov_loc': self.cov,
                'rect_types': self.rect_types,
                'maximum_elements_num': self.maximum_elements_num}


if __name__ == '__main__':
    #name_particular_rooms(path='../../data_RPLAN/floorplan_dataset/pkls',rooms={0: 1, 1: 1, 2: 1, 3: 1, 7: 1, 9: 1})
    pass
