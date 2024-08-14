import abc
from abc import ABC
from typing import List, Union, Dict
import torch
from torch import nn

from ..utils import stack_all, object_to_list, collate_all_list, detach_all, get_dic_emb_dims

try:
    from .pl_core import _BaseViewsLightning
    PL_AVAILABLE = True
except:
    PL_AVAILABLE = False

class _MVFusionCore(ABC, _BaseViewsLightning if PL_AVAILABLE else nn.Module):
    #ONLY FOR POINT-PREDICTION
    #it is based on three modules: encoders, aggregation, prediction_head
    #only one-task (prediction) and full-view available setting
    #support list and dictionary of encoders -- but transform to dict (ideally it should be always a dict)
    #support list and dictionary of emb dims -- but transform to dict
    def __init__(self,
                 view_encoders: Dict[str,nn.Module],  #require that it contains get_output_size() ..
                 merge_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_function = None,
                 maug = None, #missing as augmentation component
                 maug_args = {}, #arguments for missing augmentation
                 **kwargs
                 ):
        super(_MVFusionCore, self).__init__(**kwargs)
        if len(view_encoders) == 0:
            raise Exception("you have to give a encoder models (nn.Module), currently view_encoders=[] or {}")
        if type(prediction_head) == type(None):
            raise Exception("you need to define a prediction_head")
        if type(merge_module) == type(None):
            raise Exception("you need to define a merge_module")

        self.views_encoder = nn.ModuleDict(view_encoders)
        self.view_names = list(self.views_encoder.keys())
        self.merge_module = merge_module
        self.prediction_head = prediction_head
        
        #for training
        self.loss_function = loss_function
        self.maug = maug 
        self.maug_args = maug_args
        
        self.set_missing_info() 

    def get_encoders(self):
        return self.views_encoder
    
    def get_head(self):
        return self.prediction_head
    
    def get_embeddings_size(self):
        return get_dic_emb_dims(self.views_encoder)

    def set_missing_info(self, name:str="impute", where:str ="", value_fill=None, **kwargs):
        if name =="impute": 
            where = "input" if where == "" else where 
            value_fill = 0.0 if type(value_fill) == type(None) else value_fill
        elif name == "adapt": #for case of adapt
            where = "feature" if where =="" else where #default value: input
            value_fill = torch.nan if type(value_fill) == type(None) else value_fill 
        self.missing_method = {"name": name, "where": where, "value_fill": value_fill}

    def prepare_batch(self, batch: dict, return_target=True) -> list:
        views_data, views_target = batch["views"], batch["target"]

        if type(views_data) == list:
            if "view_names" in batch:
                if len(batch["view_names"]) != 0:
                    views_to_match = batch["view_names"]
            else:
                views_to_match = self.view_names #assuming a order list with view names based on the construction of the class
            views_dict = {views_to_match[i]: value for i, value in enumerate(views_data) }
        elif type(views_data) == dict:
            views_dict = views_data
        else:
            raise Exception("views in batch should be a List or Dict")

        if return_target:
            if type(self.loss_function) == torch.nn.CrossEntropyLoss:
                views_target = torch.squeeze(views_target).to(torch.int64)
            else:
                views_target = views_target.to(torch.float32)
        else:
            views_target = None
        return views_dict, views_target

    def transform(self,
            loader: torch.utils.data.DataLoader,
            intermediate=True,
            out_norm=False,
            device:str="",
            args_forward:dict = {},
            **kwargs
            ) -> dict:
        """
        function to get predictions from model  -- inference or testing

        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed views

        #return numpy arrays based on dictionary
        """
        device_used = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "" else device)

        missing_forward = True #flag to use random forward (based on percentage) in testing cases
        self.eval() #set batchnorm and dropout off
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views_dict, _ = self.prepare_batch(batch, return_target=False)
                for view_name in views_dict:
                    views_dict[view_name] = views_dict[view_name].to(device_used)
                    
                if missing_forward:
                    outputs_ = self(views_dict, intermediate=intermediate, out_norm=out_norm, **args_forward)
                else:
                    outputs_ = self(views_dict, intermediate=intermediate, out_norm=out_norm)
                missing_forward = True

                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_) #to start append values
                else:
                    collate_all_list(outputs, outputs_) #add to list in cpu
        self.train()
        return stack_all(outputs) #stack with numpy in cpu
    
    def apply_softmax(self, y: torch.Tensor) -> torch.Tensor:
        return nn.Softmax(dim=-1)(y)
    
    @abc.abstractmethod
    def forward_encoders(self):
        pass

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def loss_batch(self): #only for Pytorchlightning version
        pass
