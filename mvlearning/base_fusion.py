import torch, copy
from torch import nn
from typing import List, Union, Dict

from .core.fusion import _MVFusionCore
from .missing.simulate import augment_random_missing, augment_randomlist_missing

class MVFusion(_MVFusionCore):
    def __init__(self,
                 view_encoders: Dict[str,nn.Module], 
                 merge_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_function = None,
                 *args,
                 **kwargs
                 ):
        
        super(MVFusion, self).__init__(view_encoders=view_encoders, merge_module=merge_module, prediction_head=prediction_head, loss_function=loss_function, *args, **kwargs)

    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            inference_views: list = [],
            missing_method: dict = {},
            forward_only_representation: bool = False, 
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        model_views = list(views.keys()) if forward_only_representation else self.view_names 
        inference_views = model_views if len(inference_views) == 0 else inference_views

        zs_views = {}
        for v_name in model_views:
            forward_f = True  #just a flag when no forward for missing views
            if v_name in inference_views and v_name in views: #standard full-view scenario
                data_forward = views[v_name]
            else: 
                missing_method = self.missing_method if missing_method == {} else missing_method

                if missing_method.get("where") == "input": #Default: fill when view not in testing forward or view is missing
                    data_forward = torch.ones_like(views[v_name])*missing_method["value_fill"] 

                elif missing_method.get("where") == "feature": #avoid forward and fill at feature-level
                    forward_f = False
                    value_fill = torch.nan if missing_method["value_fill"] == "nan" else missing_method["value_fill"]
                    zs_views[v_name] = torch.ones(self.views_encoder[v_name].get_output_size(), device=self.device).repeat(
                                                                                                        list(views.values())[0].shape[0], 1)*value_fill
                else:
                    raise Exception("Inference with few number of views (missing) but no missing method *where* was indicated in the arguments")

            if forward_f:
                zs_views[v_name] = self.views_encoder[v_name](data_forward)
        return {"views:rep": zs_views}

    def forward(self,
            views: Dict[str, torch.Tensor],
            intermediate:bool = True,
            out_norm:bool=False,
            inference_views: list = [],
            missing_method: dict = {}, 
            forward_only_representation: bool = False, 
            forward_from_representation: bool = False, 
            ) -> Dict[str, torch.Tensor]:
        #encoder forward
        if forward_from_representation:
            out_zs_views = {"views:rep": views}
        else:
            out_zs_views = self.forward_encoders(views, inference_views=inference_views, missing_method=missing_method ,forward_only_representation=forward_only_representation) 
        if forward_only_representation:
            return out_zs_views
        
        #merge function forward
        if len(inference_views) != 0 and missing_method.get("name") == "adapt":  
            views_available_ohv = torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
        else:
            views_available_ohv = []
        views_data = [ out_zs_views["views:rep"][v] for v in sorted(out_zs_views["views:rep"].keys())] # this ensures that given the same views passed for training, this list will have a consistent order
        out_z_e = self.merge_module(views_data, views_available=views_available_ohv)
       
        #prediciton head forward
        out_y = self.prediction_head(out_z_e["joint_rep"])
        return_dic = {"prediction": self.apply_softmax(out_y) if out_norm else out_y }
        
        if intermediate:
            return dict( **return_dic, **out_zs_views, **out_z_e)
        else:
            return return_dic

    def loss_batch(self, batch: dict) -> dict:
        views_dict, views_target = self.prepare_batch(batch)
        if self.maug.lower() == "sensd" and self.training:
            inference_views = augment_random_missing(self.view_names, drop_ratio=self.maug_args.get("drop_ratio", 0.5))
            out_dic = self(views_dict, inference_views=inference_views, missing_method=self.missing_method)
        elif self.maug.lower() == "sensd" and self.training:
            inference_views = augment_randomlist_missing(self.view_names)
            out_dic = self(views_dict, inference_views=inference_views, missing_method=self.missing_method)
        else:
            out_dic = self(views_dict) 
        return {"objective": self.loss_function(out_dic["prediction"], views_target)}


class MVFusionMultiLoss(MVFusion):
    def __init__(self,
                 view_encoders: Dict[str,nn.Module],
                 merge_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_function = None,
                 multiloss_weights: Union[list, dict, float, int] = [],
                 **kwargs
                 ):
        super(MVFusionMultiLoss, self).__init__(view_encoders, merge_module, prediction_head,
            loss_function=loss_function, **kwargs)

        self.aux_predictor_base = copy.deepcopy(self.prediction_head)

        self.aux_predictor = {}
        for v_name in self.view_names:
            self.aux_predictor[v_name] = copy.deepcopy(self.aux_predictor_base)
            self.aux_predictor[v_name].load_state_dict(self.aux_predictor_base.state_dict())
            if not self.merge_module.get_info_dims().get("feature_pool"): #if not pooling, then change first layer of each predictor
                out_encoder_v_name = self.views_encoder[v_name].get_output_size()
                self.aux_predictor[v_name].update_first_layer(input_features = out_encoder_v_name)
        self.aux_predictor = nn.ModuleDict(self.aux_predictor)

        if type(multiloss_weights) == list:
            if len(multiloss_weights) == 0:
                self.multiloss_weights = {v_name: 1 for v_name in self.view_names}
            elif len(multiloss_weights) == len(self.view_names):
                self.multiloss_weights = {v_name: multiloss_weights[i] for i, v_name in enumerate(self.view_names)} #assuming orderer list
        elif type(multiloss_weights) == dict:
            self.multiloss_weights = multiloss_weights
        else: #int-- same value for all
            self.multiloss_weights = {v_name: multiloss_weights for v_name in self.view_names}

    def forward(self, views: Dict[str, torch.Tensor], intermediate = True, out_norm=False, **kwargs):
        out_dic = super(MVFusionMultiLoss, self).forward(views, intermediate = True, out_norm=out_norm, **kwargs)

        out_y_zs = {}
        for v_name in self.view_names:
            out_y_zs[v_name] = self.aux_predictor[v_name]( out_dic["views:rep"][v_name])
        out_dic["views:prediction"] = out_y_zs

        if out_norm:
            for v in out_dic: #
                if ":prediction" in v:
                    for key, value in out_dic[v].items():
                         out_dic[v][key] = self.apply_softmax(value)
        return out_dic

    def loss_batch(self, batch: dict):
        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict)
        y_x = out_dic["prediction"]
        yi_xi = out_dic["views:prediction"]
        loss_main = self.loss_function(y_x, views_target)

        loss_dic = { }
        loss_aux = 0
        for v_name in self.view_names: #
            if self.multiloss_weights[v_name] != 0:
                loss_dic["loss"+v_name] = self.multiloss_weights[v_name]*self.loss_function(yi_xi[v_name], views_target)
                loss_aux += loss_dic["loss"+v_name]
        return {"objective": loss_main+loss_aux/len(self.view_names),
                "lossmain":loss_main, "lossaux":loss_aux, **loss_dic}


class HybridFusion(MVFusionMultiLoss): #feature+decision
    def __init__(self,
                 view_encoders: Dict[str,nn.Module],
                 merge_module_feat: nn.Module,
                 prediction_head: nn.Module,
                 loss_function = None,
                 merge_module_deci: nn.Module = None,
                 multiloss_weights: Union[list, dict, float, int] = 0,
                 **kwargs
                 ):
        super(HybridFusion, self).__init__(view_encoders, merge_module_feat, prediction_head,
            loss_function=loss_function, multiloss_weights=multiloss_weights, **kwargs)
        if merge_module_deci is not None:
            self.merge_module_deci = merge_module_deci
        else:
            self.merge_module_deci = lambda x: {"joint_rep": torch.mean( torch.stack(x, 1), axis=1 )}

    def forward(self, views: Dict[str, torch.Tensor], intermediate = True, out_norm=False, **kwargs):
        out_dic = super(HybridFusion, self).forward(views, intermediate = True, out_norm=out_norm, **kwargs)
        out_dic["fusion:prediction"] = {"feat": out_dic.pop("prediction") }

        out_y_zs = {}
        for v_name in self.view_names:
            out_y_zs[v_name] = self.aux_predictor[v_name]( out_dic["views:rep"][v_name])
        out_dic["views:prediction"] = out_y_zs
        aux_out = self.merge_module_deci(list(out_y_zs.values()))
        out_dic["fusion:prediction"]["dec"] = aux_out.pop("joint_rep")
        if intermediate:
            if "att_views" in aux_out:
                aux_out["att_views_add:dec"] = aux_out.pop("att_views")
            out_dic = dict(**out_dic, **aux_out)

        out_dic["prediction"] = (out_dic["fusion:prediction"]["feat"]+out_dic["fusion:prediction"]["dec"])/2.

        if out_norm:
            for v in out_dic: #
                if ":prediction" in v:
                    for key, value in out_dic[v].items():
                         out_dic[v][key] = self.apply_softmax(value)
                elif "prediction" in v:
                    out_dic[v] = self.apply_softmax(out_dic[v])
        return out_dic


class SVPool(MVFusion): #train single-view learning models in a pool, independently between each other
    def __init__(self,
                 prediction_models: Dict[str,nn.Module],
                 loss_function = None,
                 **kwargs
                 ):
        super(SVPool, self).__init__(prediction_models, nn.Identity(), nn.Identity(),
            loss_function=loss_function, **kwargs)  

    def forward(self, views: Dict[str, torch.Tensor], intermediate=True, out_norm=False):
        #forward over entire model (encoders are the prediciton models completely)
        out_y_zs = self.forward_encoders(views)

        #do not merge!

        for v_name in out_y_zs["views:rep"]:
            out_y_zs[v_name] = self.apply_softmax(out_y_zs["views:rep"][v_name]) if out_norm else out_y_zs["views:rep"][v_name]
        out_y_zs.pop('views:rep')
        return {"views:prediction": out_y_zs}

    def loss_batch(self, batch: dict):
        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict)
        yi_xi = out_dic["views:prediction"]

        loss_dic = { }
        loss_aux = 0
        for v_name in self.view_names: 
            loss_dic["loss "+v_name] = self.loss_function(yi_xi[v_name], views_target)
            loss_aux += loss_dic["loss "+v_name]
        return {"objective": loss_aux, **loss_dic}

    def get_sv_models(self):
        return self.view_prediction_heads
