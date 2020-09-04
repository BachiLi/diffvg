"""Helper modules to build our networks."""
import torch as th


class Flatten(th.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        bs = x.shape[0]
        return x.view(bs, -1)
