import pydiffvg
import torch

class LinearGradient:
    def __init__(self,
                 begin = torch.tensor([0.0, 0.0]),
                 end = torch.tensor([0.0, 0.0]),
                 offsets = torch.tensor([0.0]),
                 stop_colors = torch.tensor([0.0, 0.0, 0.0, 0.0])):
        self.begin = begin
        self.end = end
        self.offsets = offsets
        self.stop_colors = stop_colors

class RadialGradient:
    def __init__(self,
                 center = torch.tensor([0.0, 0.0]),
                 radius = torch.tensor([0.0, 0.0]),
                 offsets = torch.tensor([0.0]),
                 stop_colors = torch.tensor([0.0, 0.0, 0.0, 0.0])):
        self.center = center
        self.radius = radius
        self.offsets = offsets
        self.stop_colors = stop_colors
