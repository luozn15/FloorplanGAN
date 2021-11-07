# FloorplanGAN  
 ![banner](assets/process_p_crop.png)
#### Introduction  
**FloorplanGAN** is a study aiming at synthesis vectorized residential floorplans based on *Differentiable Rendering*, *Adversiral Generation* and *Self-Attention*.
#### Dataset  
We leverage the open source dataset *RPlan* ([http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html)), which contains 80k+ well annotated real residential floorplans in PNG format. The training set, test set and validation set are divided 8:1:1.  
[Pyportace](https://pypi.org/project/pypotrace/) is used to vectorize these bitmap, and the preprocessed data can be download [here](https://cloud.tsinghua.edu.cn/df9310261ee5846998730/).
#### Framework
![model framework](assets/framework-03.svg)

#### Installation

1.  dependency  
```
(base)$ conda create -n floorplangan python=3.8 -y
(base)$ conda activate floorplangan
(floorplangan)$ pip install -r requirement.txt
``` 
2.  path
```
ln -s path_to_preprocessed_data 
```
3.  configuration  
modify `config.yaml` to meet your hardwares.

#### Training

1.  training
```
(floorplangan)$ python main.py
``` 
2.  visualizing the training process  
``` 
(floorplangan)$ tesorboard --logidr=runs_rplan
``` 

#### Evaluation

1.  Todo
#### Contributing
1.  luozn15@qq.com

#### License
[MIT License](LICENSE)