import torch

class LinearSum_(torch.nn.Module):
    def __init__(self, ignore: bool = False):
        super(LinearSum_, self).__init__()
        self.ignore = ignore

    def forward(self, x, **kwargs): 
        if self.ignore:
            return torch.nansum(x, dim=1) 
        else:
            return torch.sum(x, dim=1) 

class UniformSum_(torch.nn.Module):
    def __init__(self, ignore: bool = False):
        super(UniformSum_, self).__init__()
        self.ignore = ignore

    def forward(self, x, **kwargs): 
        if self.ignore:
            return torch.nanmean(x, dim=1)
        else:
            return torch.mean(x, dim=1)

class Product_(torch.nn.Module):
    def __init__(self, ignore: bool = False):
        super(Product_, self).__init__()
        self.ignore = ignore

    def forward(self, x, **kwargs): 
        if self.ignore:
            x = torch.nan_to_num(x, 1)
        return torch.prod(x, dim=1)

class Maximum_(torch.nn.Module):
    def __init__(self, ignore: bool = False):
        super(Maximum_, self).__init__()
        self.ignore = ignore

    def forward(self, x, **kwargs): 
        if self.ignore:
            x = torch.nan_to_num(x, -torch.inf)
        return torch.max(x, dim=1)[0]


class Stacking_(torch.nn.Module):
    def forward(self, x, **kwargs): 
        return torch.stack(x, dim=1)

class Concatenate_(torch.nn.Module):
    def forward(self, x, **kwargs): 
        return torch.cat(x, dim=-1)