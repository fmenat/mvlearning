import torch
from torch import nn
import abc

class Base_Encoder(abc.ABC, nn.Module):
    """
    Class to add methods for common modality specific methods
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass


class Generic_Encoder(Base_Encoder):
    """
        it adds a linear layer at the end of an encoder model with possible batch normalization.
        The linear layer could be variational in some extension
    """
    def __init__(
        self,
        encoder: nn.Module,
        latent_dims: int,
        use_norm: bool = False,
        **kwargs,
    ):
        super(Generic_Encoder, self).__init__()
        self.pre_encoder = encoder

        self.latent_dims = latent_dims
        self.use_norm = use_norm
        self.linear_layer = nn.Linear(self.pre_encoder.get_output_size(), self.latent_dims) 
        self.normalization_layer = nn.LayerNorm(self.latent_dims) if self.use_norm else nn.Identity()

    def forward(self, x):
        out_forward = self.pre_encoder(x) #should return a dictionary with output data {"rep": tensor}, or a single tensor
        if type(out_forward) != dict:
            out_forward = {"rep": out_forward}
        
        return_dic = {"rep": self.normalization_layer(self.linear_layer(out_forward["rep"])) }

        return return_dic["rep"] #single tensor output
    
    def get_output_size(self):
        return self.latent_dims
