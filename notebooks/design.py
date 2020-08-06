import torch

import swyft
import swyft.more

DEVICE = set_device(gpu=True)

sims_per_round = 3000 
dim_x = 100
dim_z = 5

def simulator(z):
    return 1.0 + z + torch.randn(z.shape) * 0.1

z = swyft.sample_hypercube(sims_per_round, dim_z)
x = swyft.more.simulate(simulator, z)

warehouse = swyft.more.Warehouse(x, z)  # needs an option for no x
combinations = [(1,), (2,), (3,), (4,), (5,), (1, 2,), (3, 4,), (1, 5,)]
net = swyft.Network(xdim = ydim, zdim = NDIM, head = Head().to(DEVICE)).to(DEVICE)

# TODO think about this function, do we want getting data split in two?
# (I think so for this level of abstraction.)
train_data, valid_data = warehouse.get_dataset(percent_train=0.9)
train_loader, valid_loader = make_loaders(train_data, valid_data) 

train_loss, valid_loss, best_state_dict = swyft.more.train(net, train_loader, valid_loader, early_stopping_patience = 20)