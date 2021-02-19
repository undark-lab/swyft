import torch
import swyft


class Head_FermiV1(swyft.Module):
    def __init__(self, obs_shapes):
        super().__init__(obs_shapes=obs_shapes)
        
        self.n_features = 10

        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.conv3 = torch.nn.Conv2d(20, 40, 5)
        self.pool = torch.nn.MaxPool2d(2)
        self.l = torch.nn.Linear(160, 10)
        
    def forward(self, obs):
        x = obs['mu'].unsqueeze(1)
        nbatch = len(x)
        #x = torch.log(0.1+x)
        
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(nbatch, -1)
        x = self.l(x)

        return x
