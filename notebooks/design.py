import torch

import swyft
import swyft.more

# The goal of this is to show how the "nuts and bolts" level of swyft would work. 
# There would be another layer on top which is what the physicists interact with.

# One problem with pyrofit lensing is that the data is saved to disk, so you need to create a custom way to use that data with swyft.
# Specifically the interation part of it.

DEVICE = set_device(gpu=True)

sims_per_round = 3000 
dim_x = 100
dim_z = 5
threshold = 1e-6
rounds = 5

def simulator(z):
    return 1.0 + z + torch.randn(z.shape) * 0.1

warehouse = swyft.more.Warehouse()
combinations = [(1,), (2,), (3,), (4,), (5,), (1, 2,), (3, 4,), (1, 5,)]

z0 = torch.tensor([1,])
x0 = swyft.more.simulate(simulator, z0)

# simulate
# add to warehouse
# get data from warehouse
# learn
# append learned info to warehouse
# repeat

for r in range(rounds):
    if r == 0:
        z = swyft.sample_hypercube(sims_per_round, dim_z)
        x = swyft.more.simulate(simulator, z)
        rounds = torch.zeros(z.size(0))
    else:
        # This whole thing should probably be turned into a function.
        existing_x, existing_z, existing_round = swyft.more.apply_mask(mask, x, z, rounds)
        new_z, new_round = swyft.more.sample(sims_per_round, dim_z, mask, existing_z, existing_round)
        new_x = swyft.more.simulate(simulator, new_z)
        x = torch.cat(existing_x, new_x)
        z = torch.cat(existing_z, new_z)
        rounds = torch.cat(existing_round, new_round)

    # TODO improve this, along with how network loops over legs
    net = swyft.Network(xdim = ydim, zdim = NDIM, head = Head().to(DEVICE)).to(DEVICE)

    warehouse.append(x=x, z=z, rounds=rounds)

    # TODO think about this function, do we want getting data split in two?
    # (I think so for this level of abstraction.)
    train_data, valid_data = warehouse.get_dataset(subset_percents=[0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_data) 
    valid_loader = torch.utils.data.DataLoader(valid_data)

    train_loss, valid_loss, best_state_dict = swyft.more.train(net, train_loader, valid_loader, early_stopping_patience = 20)

    net.load_state_dict(best_state_dict)
    mask = swyft.more.get_masking_fn(net, x0, threshold)
    warehouse.append(likelihood_estimator=net)
