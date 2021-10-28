from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pickle


def name_particular_rooms(path, rooms):
    if rooms == None:
        print('rooms == None')
        return
    suffixes = '_'.join(['{}-{}'.format(k, v) for k, v in rooms.items()])
    file = os.path.join(path, '../', 'names', 'names_{}.pkl'.format(suffixes))
    if os.path.exists(file):
        print('names_{}.pkl exists'.format(suffixes))
        return
    names = os.listdir(path)
    names_ = []
    for name in tqdm(names):
        with open(os.path.join(path, name), 'rb') as pkl_file:
            layout = pickle.load(pkl_file)

        if np.prod([len(layout[k]) == v for k, v in rooms.items()]) > 0:
            names_.append(name)
    with open(file, 'wb') as output:
        pickle.dump(names_, output)
    print('find {} layouts satisfying the rooms requirement'.format(len(names_)))
    return


def types_more_than_n(path, n=2000):
    file = os.path.join(path, '../', 'names', 'morethan_{}.pkl'.format(str(n)))
    if os.path.exists(file):
        print('morethan_{}.pkl exists'.format(str(n)))
        return
    names = os.listdir(path)
    rooms = range(10)  # (0,1)
    num_types = {}
    for name in tqdm(names):
        with open(os.path.join(path, name), 'rb') as pkl_file:
            layout = pickle.load(pkl_file)
        if '_'.join([str(len(layout[r])) for r in rooms]) in num_types.keys():
            num_types['_'.join([str(len(layout[r])) for r in rooms])] += 1
        else:
            num_types['_'.join([str(len(layout[r])) for r in rooms])] = 1
    num_types = pd.DataFrame.from_dict(num_types, orient='index')
    num_types = num_types.sort_values(0, ascending=True)
    types = list(num_types[num_types > 2000].dropna().index)

    names_ = []
    for name in tqdm(names):
        with open(os.path.join(path, name), 'rb') as pkl_file:
            layout = pickle.load(pkl_file)
        if '_'.join([str(len(layout[r])) for r in rooms]) in types:
            names_.append(name)
    with open(file, 'wb') as output:
        pickle.dump(names_, output)
    print('find {} types - {} layouts out of {} in total'.format(len(types),
          len(names_), len(names)))
    return


def bounds_check(generated):
    loc = generated[:, :, -4:]
    '''xc,yc,area_root,w = loc[:,:,0],loc[:,:,1],loc[:,:,2],loc[:,:,3]
    h = area_root**2/w
    try:
        h[h != h] = 0
    except:
        pass'''
    xc, yc, w, h = loc[:, :, 0], loc[:, :, 1], loc[:, :, 2], loc[:, :, 3]

    x0 = xc-0.5*w
    x1 = xc+0.5*w
    y0 = yc-0.5*h
    y1 = yc+0.5*h

    def loss_(c):
        return (F.relu(c) - c + F.relu(1-c) - (1-c)).sum()
    loss = loss_(x0)+loss_(y0)+loss_(x1)+loss_(y1)
    return loss


def negative_wh_check(generated):
    w = generated[:, :, -2]
    h = generated[:, :, -1]

    def loss_(c):
        return (F.relu(c) - c).sum()
    loss = loss_(w)+loss_(h)
    return loss


def get_figure(render):
    color_table = {
        0: (210, 121, 98),  # Living room
        1: (238, 216, 98),  # Master room
        2: (83, 103, 52),  # Kitchen
        3: (118, 142, 168),  # Bathroom
        4: (82, 79, 115),  # Dining room
        5: (227, 152, 68),  # Child room
        6: (145, 177, 101),  # Study room
        7: (59, 105, 138),  # Second room
        8: (36, 35, 42),  # Guest room
        9: (221, 209, 212),  # Balcony
    }
    try:
        render = render.detach().cpu().numpy()
    except:
        pass

    rendered_size = render.shape[-1]
    batch_size = render.shape[0]
    num_channel = render.shape[1]
    fig = plt.figure(figsize=(18, 18))
    for i in range(batch_size):
        img_stack = []
        for j in range(num_channel):
            num = i*(num_channel+1)+j+1
            ax = fig.add_subplot(batch_size, num_channel+1, num)
            #img = render[i,j,:,:].view(rendered_size,rendered_size,1).expand(rendered_size,rendered_size,3)
            img = np.tile(render[i, j, :, :, np.newaxis], (1, 1, 3))
            img *= np.array(color_table[j])
            img = img.astype('int')
            img_stack.append(img)
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if i == 0:
                ax.set_title(str(j))
        ax = fig.add_subplot(batch_size, num_channel+1, num+1)
        img_stack = np.array(img_stack)
        ax.imshow(np.max(img_stack, axis=0), cmap='gray', vmin=0, vmax=255)
        if i == 0:
            ax.set_title('floorplan')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


def draw_table(df):
    if type(df) == torch.Tensor:
        df = df.cpu().detach().numpy()
    table = plt.figure(figsize=(16, 2))
    ax = table.add_subplot(111)
    ax.table(np.around(df, decimals=6), loc='center')
    ax.axis('off')
    return table


def geo_covariance_loss(generated, real):
    covariance_generated = torch.matmul(
        generated[:, :, -4:], generated[:, :, -4:].permute(0, 2, 1))
    covariance_real = torch.matmul(
        real[:, :, -4:], real[:, :, -4:].permute(0, 2, 1))
    crit = nn.BCEWithLogitsLoss()
    loss = crit(covariance_generated, covariance_real)
    return loss


def ratio_loss(real_images, dataset):
    weight = real_images[0][:, :, :-4].sum(axis=-1) > 0.5
    ratio = real_images[0][:, :, -2].div(real_images[0][:, :, -1])
    x = weight*ratio


def print_grad(net):
    for name, weight in net.named_parameters():
        # print("weight:", weight) # 打印权重，看是否在变化
        if weight.requires_grad:
            # print("weight:", weight.grad) # 打印梯度，看是否丢失
            # 直接打印梯度会出现太多输出，可以选择打印梯度的均值、极值，但如果梯度为None会报错
            print("{:50} grad:".format(name), weight.grad.mean())
