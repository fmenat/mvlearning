from torch import nn
from typing import List, Union, Dict

from .base_fusion import MVFusionMissing, MVFusionMissingMultiLoss, SVPool, HybridFusion
from .fusion_module import FusionModuleMissing
from .utils import Lambda, get_dic_emb_dims

class InputFusion(MVFusionMissing):
    def __init__(self,
                 predictive_model,
                 view_names: list,
                 fusion_module: dict = {},
                 loss_function= None,
                 ):
        """
        Arguments
        ---
            * predictive_model: Torch model that predicts the target
            * view_names: A list with all the view names
            * fusion_module: (Optional) If want to use an alternative fusion outside concatenation
            * loss_function: Torch loss function module
        """
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "concat", "emb_dims": [0 for _ in view_names] }
            fusion_module = FusionModuleMissing(**fusion_module)
        fake_view_encoders = {}
        for v in view_names:
            aux = nn.Identity()
            #aux.get_output_size = lambda : v
            fake_view_encoders[v] = aux
        super(InputFusion, self).__init__(fake_view_encoders, fusion_module, predictive_model,
            loss_function=loss_function)

class FeatureFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_function= None,
                 ):
        """
        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * fusion_module: Merge function used, such as concatenation, average, weighted sum.
            * predictive_model: Single torch model that predicts the target            
            * loss_function: Torch loss function module
        """
        super(FeatureFusion, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_function=loss_function)

class FeatureFusionMultiLoss(MVFusionMissingMultiLoss):
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_function= None,
                 multiloss_weights = [],
                 ):
        """
        Extension of Feature Fusion that includes MultiLoss

        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * fusion_module: Merge function used, such as concatenation, average, weighted sum.
            * predictive_model: Single torch model that predicts the target            
            * loss_function: Torch loss function module
            * multiloss_weights: (Optional) a single value or a dictionary with the values in the loss for each view
        """
        super(FeatureFusionMultiLoss, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_function=loss_function,multiloss_weights=multiloss_weights)

class DecisionFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 fusion_module: dict = {},
                 loss_function= None,
                 ):
        """
        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * fusion_module: (Optional) If want to use an alternative fusion outside averaging
            * loss_function: Torch loss function module
        """
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "avg", "adaptive":False, "emb_dims": get_dic_emb_dims(view_encoders) }
            fusion_module = FusionModuleMissing(**fusion_module)
        super(DecisionFusion, self).__init__(view_encoders, fusion_module, Lambda(lambda x: x["rep"] if type(x) == dict else x),
            loss_function=loss_function)

class DecisionFusionMultiLoss(MVFusionMissingMultiLoss):
    def __init__(self,
                 view_encoders,
                 fusion_module: dict = {},
                 loss_function= None,
                 multiloss_weights = [],
                 ):
        """
        Extension of Decision Fusion that includes MultiLoss

        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * fusion_module: (Optional) If want to use an alternative fusion outside averaging
            * loss_function: Torch loss function module
            * multiloss_weights: (Optional) a single value or a dictionary with the values in the loss for each view
        """
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "avg", "adaptive":False, "emb_dims": get_dic_emb_dims(view_encoders)}
            fusion_module = FusionModuleMissing(**fusion_module)
        super(DecisionFusionMultiLoss, self).__init__(view_encoders, fusion_module,  Lambda(lambda x: x["rep"] if type(x) == dict else x),
            loss_function=loss_function, multiloss_weights=multiloss_weights)

class HybridFusion_FD(HybridFusion):
    def __init__(self,
                 view_encoders,
                 fusion_module_feat: nn.Module,
                 predictive_model: nn.Module,
                 loss_function= None,
                 fusion_module_deci: nn.Module = None,
                 ):
        """
        Arguments
        ---
            * view_encoders: A dictionary with a Torch model that generate the embedding of each view.
            * fusion_module: Merge function used, such as concatenation, average, weighted sum.
            * predictive_model: Single torch model that predicts the target            
            * loss_function: Torch loss function module
            * fusion_module_deci: (Optional) Merge function used in the decision level, otherwise the average will be used
        """
        super(HybridFusion_FD, self).__init__(view_encoders, fusion_module_feat, predictive_model,
             loss_function=loss_function,fusion_module_deci=fusion_module_deci)

class SingleViewPool(SVPool):
    pass
