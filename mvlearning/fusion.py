from torch import nn
from typing import List, Union, Dict

from .base_fusion import MVFusionMissing, MVFusionMissingMultiLoss, SVPool, HybridFusion
from .merge_module import MergeModule
from .utils import Lambda, get_dic_emb_dims

class InputFusion(MVFusionMissing):
    def __init__(self,
                 predictive_model,
                 view_names: list,
                 merge_module: dict = {},
                 loss_function= None,
                 **kwargs
                 ):
        """
        Arguments
        ---
            * predictive_model: Torch model that predicts the target
            * view_names: A list with all the view names
            * merge_module: (Optional) If want to use an alternative fusion outside concatenation
            * loss_function: Torch loss function module
        """
        if type(merge_module) == dict:
            if len(merge_module) == 0:
                merge_module = {"mode": "concat", "emb_dims": [0 for _ in view_names] }
            merge_module = MergeModule(**merge_module)
        fake_view_encoders = {}
        for v in view_names:
            aux = nn.Identity()
            #aux.get_output_size = lambda : v
            fake_view_encoders[v] = aux
        super(InputFusion, self).__init__(fake_view_encoders, merge_module, predictive_model,
            loss_function=loss_function, **kwargs)

class FeatureFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 merge_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_function= None,
                 **kwargs
                 ):
        """
        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * merge_module: Merge function used, such as concatenation, average, weighted sum.
            * predictive_model: Single torch model that predicts the target            
            * loss_function: Torch loss function module
        """
        super(FeatureFusion, self).__init__(view_encoders, merge_module, predictive_model,
             loss_function=loss_function, **kwargs)

class FeatureFusionMultiLoss(MVFusionMissingMultiLoss):
    def __init__(self,
                 view_encoders,
                 merge_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_function= None,
                 multiloss_weights = [],
                 **kwargs
                 ):
        """
        Extension of Feature Fusion that includes MultiLoss

        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * merge_module: Merge function used, such as concatenation, average, weighted sum.
            * predictive_model: Single torch model that predicts the target            
            * loss_function: Torch loss function module
            * multiloss_weights: (Optional) a single value or a dictionary with the values in the loss for each view
        """
        super(FeatureFusionMultiLoss, self).__init__(view_encoders, merge_module, predictive_model,
             loss_function=loss_function,multiloss_weights=multiloss_weights, **kwargs)

class DecisionFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 merge_module: dict = {},
                 loss_function= None,
                 **kwargs
                 ):
        """
        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * merge_module: (Optional) If want to use an alternative fusion outside averaging
            * loss_function: Torch loss function module
        """
        if type(merge_module) == dict:
            if len(merge_module) == 0:
                merge_module = {"mode": "avg", "adaptive":False, "emb_dims": get_dic_emb_dims(view_encoders) }
            merge_module = MergeModule(**merge_module)
        super(DecisionFusion, self).__init__(view_encoders, merge_module, Lambda(lambda x: x["rep"] if type(x) == dict else x),
            loss_function=loss_function, **kwargs)

class DecisionFusionMultiLoss(MVFusionMissingMultiLoss):
    def __init__(self,
                 view_encoders,
                 merge_module: dict = {},
                 loss_function= None,
                 multiloss_weights = [],
                 **kwargs
                 ):
        """
        Extension of Decision Fusion that includes MultiLoss

        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * merge_module: (Optional) If want to use an alternative fusion outside averaging
            * loss_function: Torch loss function module
            * multiloss_weights: (Optional) a single value or a dictionary with the values in the loss for each view
        """
        if type(merge_module) == dict:
            if len(merge_module) == 0:
                merge_module = {"mode": "avg", "adaptive":False, "emb_dims": get_dic_emb_dims(view_encoders)}
            merge_module = MergeModule(**merge_module)
        super(DecisionFusionMultiLoss, self).__init__(view_encoders, merge_module,  Lambda(lambda x: x["rep"] if type(x) == dict else x),
            loss_function=loss_function, multiloss_weights=multiloss_weights, **kwargs)

class HybridFusion_FD(HybridFusion):
    def __init__(self,
                 view_encoders,
                 merge_module_feat: nn.Module,
                 predictive_model: nn.Module,
                 loss_function= None,
                 merge_module_deci: nn.Module = None,
                 **kwargs
                 ):
        """
        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * merge_module: Merge function used, such as concatenation, average, weighted sum.
            * predictive_model: Single torch model that predicts the target            
            * loss_function: Torch loss function module
            * merge_module_deci: (Optional) Merge function used in the decision level, otherwise the average will be used
        """
        super(HybridFusion_FD, self).__init__(view_encoders, merge_module_feat, predictive_model,
             loss_function=loss_function,merge_module_deci=merge_module_deci, **kwargs)

class SingleViewPool(SVPool):
    pass
