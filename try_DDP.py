# main.py文件
import torch
import os
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
# 新增：从外面得到local_rank参数


def train():
    print(datetime.now().strftime('%Y-%m-%d'))
    print(datetime.now().strftime('%H-%M-%S'))
    local_rank = int(os.environ["LOCAL_RANK"])
    # 构造模型
    device = torch.device("cuda", local_rank)
    model = nn.Linear(10, 10).to(device)
    # 新增：构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 前向传播
    outputs = model(torch.randn(20, 10).to(device))
    labels = torch.randn(20, 10).to(device)
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    print(outputs)
    # 后向传播
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()

    print(local_rank, 'end')


if __name__ == '__main__':
    import os
    local_rank = int(os.environ["LOCAL_RANK"])
    with torch.cuda.device(local_rank):
        torch.distributed.init_process_group(backend='nccl')
        # torch.autograd.set_detect_anomaly(True)
        train()
