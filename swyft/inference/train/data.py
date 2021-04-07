## pylint: disable=no-member, not-callable
#import torch
#from torch.utils.data import Dataset
#
#class ParamDictDataset(Dataset):
#    def __init__(self, points):
#        self.points = points
#
#    def _tensorfy(self, x):
#        return {k: torch.tensor(v).float() for k, v in x.items()}
#
#    def __len__(self):
#        return len(self.points)
#
#    def __getitem__(self, i):
#        p = self.points[i]
#        return dict(obs=self._tensorfy(p["obs"]), par=torch.tensor(p['par']).float())
