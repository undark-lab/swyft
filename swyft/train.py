# pylint: disable=no-member, not-callable
import math
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from .utils import combine_z


def loss_fn(network, xz, combinations=None):
    """Evaluate binary-cross-entropy loss function. Mean over batch.

    Args:
        network (nn.Module): network taking minibatch of samples and returing ratio estimator.
        xz (dict): batch of samples to train on.
        combinations (list, optional): determines posteriors that are generated.
            examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4

    Returns:
        Tensor: training loss.
    """  # TODO does the loss function depend on which distribution the z was drawn from? it does in SBI for the SNPE versions
    assert xz["x"].size(0) == xz["z"].size(0), "Number of x and z must be equal."
    assert (
        xz["x"].size(0) % 2 == 0
    ), "There must be an even number of samples in the batch for contrastive learning."
    n_batch = xz["x"].size(0)

    # Is it the removal of replacement that made it stop working?!

    # bring x into shape
    # (n_batch*2, data-shape)  - repeat twice each sample of x - there are n_batch samples
    # repetition pattern in first dimension is: [a, a, b, b, c, c, d, d, ...]
    x = xz["x"]
    x = torch.repeat_interleave(x, 2, dim=0)

    # bring z into shape
    # (n_batch*2, param-shape)  - repeat twice each sample of z - there are n_batch samples
    # repetition is alternating in first dimension: [a, b, a, b, c, d, c, d, ...]
    z = xz["z"]
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    zdim = len(z[0])
    z = z.view(n_batch // 2, -1, *z.shape[-1:])
    z = torch.repeat_interleave(z, 2, dim=0)
    z = z.view(n_batch * 2, -1, *z.shape[-1:])

    # call network
    lnL = network(x, z)
    lnL = lnL.view(n_batch // 2, 4, zdim)

    # Evaluate cross-entropy loss
    # loss =
    # -ln( exp(lnL(x_a, z_a))/(1+exp(lnL(x_a, z_a))) )
    # -ln( exp(lnL(x_b, z_b))/(1+exp(lnL(x_b, z_b))) )
    # -ln( 1/(1+exp(lnL(x_a, z_b))) )
    # -ln( 1/(1+exp(lnL(x_b, z_a))) )
    loss = -torch.nn.functional.logsigmoid(lnL[:, 0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 2])
    loss += -torch.nn.functional.logsigmoid(lnL[:, 3])
    loss = loss.sum() / (n_batch // 2)

    return loss


# We have the posterior exactly because our proir is known and flat. Flip bayes theorem, we have the likelihood ratio.
# Consider that the variance of the loss from different legs causes some losses to have high coefficients in front of them.
def train(
    network,
    train_loader,
    validation_loader,
    early_stopping_patience,
    max_epochs=None,
    lr=1e-3,
    combinations=None,
    device=None,
    non_blocking=True,
):
    """Network training loop.

    Args:
        network (nn.Module): network for ratio estimation.
        train_loader (DataLoader): DataLoader of samples.
        validation_loader (DataLoader): DataLoader of samples.
        max_epochs (int): Number of epochs.
        lr (float): learning rate.
        combinations (list, optional): determines posteriors that are generated.
            examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4
        device (str, device): Move batches to this device.
        non_blocking (bool): non_blocking in .to(device) expression.

    Returns:
        list: list of training losses.
    """
    # TODO consider that the user might want other training stats, like number of correct samples for example
    def do_epoch(loader: torch.utils.data.dataloader.DataLoader, train: bool):
        accumulated_loss = 0
        training_context = nullcontext() if train else torch.no_grad()
        with training_context:
            for batch in loader:
                optimizer.zero_grad()
                if device is not None:
                    batch = {
                        k: v.to(device, non_blocking=non_blocking)
                        for k, v in batch.items()
                    }
                loss = loss_fn(network, batch, combinations=combinations)
                if train:
                    loss.backward()
                    optimizer.step()
                accumulated_loss += loss.detach().cpu().numpy().item()
        return accumulated_loss

    max_epochs = 2 ** 31 - 1 if max_epochs is None else max_epochs
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    n_train_batches = len(train_loader)
    n_validation_batches = len(validation_loader)

    train_losses, validation_losses = [], []
    epoch, fruitless_epoch, min_loss = 0, 0, float("Inf")
    while epoch < max_epochs and fruitless_epoch < early_stopping_patience:
        # print("Epoch:", epoch, end = "\r")
        network.train()
        train_loss = do_epoch(train_loader, True)
        train_losses.append(train_loss / n_train_batches)

        network.eval()
        validation_loss = do_epoch(validation_loader, False)
        print("Validation loss:", validation_loss)
        validation_losses.append(validation_loss / n_validation_batches)

        epoch += 1
        if epoch == 0 or min_loss > validation_loss:
            fruitless_epoch = 0
            min_loss = validation_loss
            best_state_dict = deepcopy(network.state_dict())
        else:
            fruitless_epoch += 1

    print("Total epochs:", epoch)
    # print("Validation losses:", validation_losses)

    return train_losses, validation_losses, best_state_dict


def trainloop(
    net,
    dataset,
    combinations=None,
    nbatch=32,
    nworkers=4,
    max_epochs=50,
    early_stopping_patience=1,
    device="cpu",
    lr_schedule=[1e-3, 1e-4, 1e-5],
    nl_schedule=[1.0, 1.0, 1.0],
):
    print("Start training")
    nvalid = 512
    ntrain = len(dataset) - nvalid
    dataset_train, dataset_valid = torch.utils.data.random_split(
        dataset, [ntrain, nvalid]
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=nbatch,
        num_workers=nworkers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=nbatch,
        num_workers=nworkers,
        pin_memory=True,
        drop_last=True,
    )
    # Train!

    train_loss, valid_loss = [], []
    for i, lr in enumerate(lr_schedule):
        print(f"LR iteration {i}")
        # dataset.set_noiselevel(nl_schedule[i])
        tl, vl, sd = train(
            net,
            train_loader,
            valid_loader,
            early_stopping_patience=early_stopping_patience,
            lr=lr,
            max_epochs=max_epochs,
            device=device,
            combinations=combinations,
        )
        vl_minimum = min(vl)
        vl_min_idx = vl.index(vl_minimum)
        train_loss.append(tl[: vl_min_idx + 1])
        valid_loss.append(vl[: vl_min_idx + 1])
        net.load_state_dict(sd)


def combine(y, z):
    """Combines data vectors y and parameter vectors z.

    z : (..., pnum, pdim)
    y : (..., ydim)

    returns: (..., pnum, ydim + pdim)

    """
    y = y.unsqueeze(-2)  # (..., 1, ydim)
    y = y.expand(*z.shape[:-1], *y.shape[-1:])  # (..., pnum, ydim)
    return torch.cat([y, z], -1)


def get_norms(xz, combinations=None, N=300):
    irand = np.random.choice(len(xz), N)
    x = [xz[i]["x"] for i in irand]
    z = [xz[i]["z"] for i in irand]
    x_mean = sum(x) / len(x)
    z_mean = sum(z) / len(z)
    x_var = sum([(x[i] - x_mean) ** 2 for i in range(len(x))]) / len(x)
    z_var = sum([(z[i] - z_mean) ** 2 for i in range(len(z))]) / len(z)

    z_mean = combine_z(z_mean, combinations)
    z_var = combine_z(z_var, combinations)

    # print("Normalizations")
    # print("x_mean", x_mean)
    # print("x_err", x_var**0.5)
    # print("z_mean", z_mean)
    # print("z_err", z_var**0.5)

    return x_mean, x_var ** 0.5, z_mean, z_var ** 0.5


# From: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        # initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, output_size, input_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, output_size))

        # change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.matmul(self.w, x).squeeze(-1) + self.b


class DenseLegs(nn.Module):
    def __init__(self, ydim, pnum, pdim=1, p=0.0, NH=256):
        super().__init__()
        self.fc1 = LinearWithChannel(ydim + pdim, NH, pnum)
        self.fc2 = LinearWithChannel(NH, NH, pnum)
        self.fc3 = LinearWithChannel(NH, NH, pnum)
        self.fc4 = LinearWithChannel(NH, 1, pnum)
        self.drop = nn.Dropout(p=p)

        self.af = torch.relu

        # swish activation function for smooth posteriors
        self.af2 = lambda x: x * torch.sigmoid(x * 10.0)

    def forward(self, y, z):
        x = combine(y, z)
        x = self.af(self.fc1(x))
        x = self.drop(x)
        x = self.af(self.fc2(x))
        x = self.drop(x)
        x = self.af(self.fc3(x))
        x = self.fc4(x).squeeze(-1)
        return x


class Network(nn.Module):
    def __init__(self, ydim, pnum, pdim=1, head=None, p=0.0, datanorms=None):
        """Base network combining z-independent head and parallel tail.

        :param ydim: Number of data dimensions going into DenseLeg network
        :param pnum: Number of posteriors to estimate
        :param pdim: Dimensionality of posteriors
        :param head: Head network, z-independent
        :type head: `torch.nn.Module`, optional

        The forward method of the `head` network takes data `x` as input, and
        returns intermediate state `y`.
        """
        super().__init__()
        self.head = head
        self.legs = DenseLegs(ydim, pnum, pdim=pdim, p=p)

        # Set datascaling
        if datanorms is None:
            datanorms = [
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(0.5),
                torch.tensor(0.5),
            ]
        self._set_datanorms(*datanorms)

    def _set_datanorms(self, x_mean, x_std, z_mean, z_std):
        self.x_loc = torch.nn.Parameter(x_mean)
        self.x_scale = torch.nn.Parameter(x_std)
        self.z_loc = torch.nn.Parameter(z_mean)
        self.z_scale = torch.nn.Parameter(z_std)

    def forward(self, x, z):
        x = (x - self.x_loc) / self.x_scale
        z = (z - self.z_loc) / self.z_scale

        if self.head is not None:
            y = self.head(x)
        else:
            y = x  # Use 1-dim data vector as features

        out = self.legs(y, z)
        return out


if __name__ == "__main__":
    pass
