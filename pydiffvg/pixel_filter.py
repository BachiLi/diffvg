import torch
import pydiffvg

class PixelFilter:
    def __init__(self,
                 type,
                 radius = torch.tensor(0.5)):
        self.type = type
        self.radius = radius
