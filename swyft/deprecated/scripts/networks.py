import swyft.lightning as sl
import numpy as np
import torch
import swyft

class UNet(sl.SwyftModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.online_z_score = swyft.networks.OnlineDictStandardizingLayer(dict(data = (256, 256)))
        self.L = torch.nn.Sequential(
                torch.nn.LazyConv2d(16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.LazyLinear(256)
                )
        self.classifier = sl.RatioEstimatorMLP1d(256, 5, hidden_features = 256)

    def forward(self, x, z):
        data = self.online_z_score(dict(data = 2.0*x['data']))['data']
        f = self.L(data.unsqueeze(1))
        z = z['z']
        ratios_z = self.classifier(f, z)
        return dict(z = ratios_z)

class CNN(sl.SwyftModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.online_z_score = swyft.networks.OnlineDictStandardizingLayer(dict(data = (256, 256)))
        self.CNN = torch.nn.Sequential(
            torch.nn.LazyConv2d(32, 3), torch.nn.ReLU(),
            torch.nn.LazyConv2d(32, 3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.LazyConv2d(64, 3), torch.nn.ReLU(),
            torch.nn.LazyConv2d(64, 3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.LazyConv2d(128, 3), torch.nn.ReLU(),
            torch.nn.LazyConv2d(128, 3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.LazyConv2d(256, 3), torch.nn.ReLU(),
            torch.nn.LazyConv2d(256, 3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(256),
        )
        self.classifier1 = sl.RatioEstimatorMLP1d(256, 5, hidden_features = 256)
        #marginals = ((0, 1), (2, 3))
        #self.classifier2 = sl.RatioEstimatorMLPnd(16, marginals, hidden_features = 256)

    def forward(self, x, z):
        # Digesting x
        x = dict(data = x['data']*1.)
        #x = x['data']*1.
        data = self.online_z_score(x)['data']
        f = self.CNN(data.unsqueeze(1)).squeeze(1)
        z = z['z']
        
        ratios_z = self.classifier1(f, z)
        #ratios_zz = self.classifier2(x, z['z'])
        return dict(z = ratios_z)#, zz = ratios_zz)


def get_network(name):
    if name == "CNN":
        return CNN
    else:
        raise KeyError
