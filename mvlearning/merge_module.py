import torch
import numpy as np
from typing import List, Union, Dict

from .core.fusion_layers import LinearSum_,UniformSum_,Product_,Maximum_,Stacking_,Concatenate_

POOL_FUNC_NAMES = ["sum", "avg","mean","linearsum", "prod", "mul" ,"max", "pool", "weightedsum"]
STACK_FUNC_NAMES = ["concat" ,"stack", "concatenate", "stacking", "concatenating", "cat"]


class MergeModule(torch.nn.Module):
    def __init__(self, emb_dims: List[int], mode: str, adaptive: bool=False, features: bool=False, activation_fun="softmax",**kwargs):
        super(MergeModule, self).__init__()
        self.mode = mode
        self.adaptive = adaptive
        self.features = features #only used when adaptive=True
        self.activation_fun = activation_fun #only used when adaptive=True
        if type(emb_dims) == dict: #assuming a orderer list based on dictionary
            emb_dims = list(emb_dims.values())
        self.emb_dims = emb_dims
        self.N_views = len(emb_dims)
        self.joint_dim, self.feature_pool = self.get_dim_agg()
        self.check_valid_args()

        if self.feature_pool:
            self.stacker_function = Stacking_()

        if self.mode in STACK_FUNC_NAMES:
            self.concater_function = Concatenate_()

        elif self.mode.split("_")[0] in ["avg","mean","uniformsum"]:
            self.pooler_function = UniformSum_(ignore = self.mode.split("_")[-1] == "ignore" )

        elif self.mode.split("_")[0] in ["sum","add","linearsum"]:
            self.pooler_function = LinearSum_(ignore = self.mode.split("_")[-1] == "ignore")

        elif self.mode.split("_")[0] in ["prod", "mul"]:
            self.pooler_function = Product_(ignore = self.mode.split("_")[-1] == "ignore")

        elif self.mode.split("_")[0] in ["max", "pool"]:
            self.pooler_function = Maximum_(ignore = self.mode.split("_")[-1] == "ignore")
        
        else:
            raise ValueError(f'Invalid value for mode: {self.mode}. Valid values: {POOL_FUNC_NAMES+STACK_FUNC_NAMES}')

        if self.adaptive:
            if self.mode in STACK_FUNC_NAMES:
                forward_input_dim = sum(self.emb_dims)
            else:
                forward_input_dim = self.joint_dim
            forward_output_dim = self.joint_dim*self.N_views if self.features else self.N_views

            self.attention_function = torch.nn.Linear(forward_input_dim, forward_output_dim)

    def get_dim_agg(self):
        if self.adaptive or (self.mode.split("_")[0] not in STACK_FUNC_NAMES):
            fusion_dim = self.emb_dims[0]
            feature_pool = True
        else:
            fusion_dim = sum(self.emb_dims)
            feature_pool = False
        return fusion_dim, feature_pool

    def check_valid_args(self):
        if len(np.unique(self.emb_dims)) != 1:
            if self.adaptive:
                raise Exception("Cannot set adaptive=True when the number of features in embedding are not the same")
            if self.mode.split("_")[0] in POOL_FUNC_NAMES:
                raise Exception("Cannot set pooling aggregation when the number of features in embedding are not the same")

    def forward(self, views_emb: List[torch.Tensor], views_available = []) -> Dict[str, torch.Tensor]: #the list is always orderer based on previous models
        if self.feature_pool:
            views_stacked = self.stacker_function(views_emb)

        if self.mode in STACK_FUNC_NAMES:
            joint_emb_views = self.concater_function(views_emb)

        elif self.mode.split("_")[0] in POOL_FUNC_NAMES:
            joint_emb_views = self.pooler_function(views_stacked)
            
        if self.adaptive:
            att_views = self.attention_function(joint_emb_views)

            if self.features:
                att_views = torch.reshape(att_views, (att_views.shape[0], self.N_views, self.joint_dim))
            else:
                att_views = att_views[:,:,None]

            if len(views_available) != 0:  #missing case, masking attention
                if len(views_available.shape) < len(att_views.shape):
                    if len(views_available.shape) == 2:
                        views_available = (views_available[:, :, None]).repeat(1, 1, att_views.shape[-1])           
                    elif len(views_available.shape) ==1:
                        views_available = (views_available[None, :, None]).repeat(att_views.shape[0], 1, att_views.shape[-1])     
                att_views[views_available == 0] = np.log(1e-20) #-torch.inf

            if self.activation_fun.lower() == "softmax":
                att_views = torch.nn.Softmax(dim=-1)(att_views)
            elif self.activation_fun.lower() == "tanh":
                att_views = torch.nn.Tanh()(att_views)
            elif self.activation_fun.lower() == "sigmoid":
                att_views = torch.nn.Sigmoid()(att_views)
            joint_emb_views = torch.sum(views_stacked*att_views, dim=1)

        dic_return = {"joint_rep": joint_emb_views}
        if self.adaptive:
            dic_return["att_views"] = att_views
        return dic_return

    def get_info_dims(self):
        return { 
            "emb_dims":self.emb_dims, 
            "joint_dim":self.joint_dim, 
            "feature_pool": self.feature_pool
            }

    def get_joint_dim(self):
        return self.joint_dim
