import torch
from torch import nn

class EP_PKVAE(nn.Module):
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB


    def editing_params(self,source_param,target_param,source_keypoint): 
        condition = torch.cat((source_param-target_param,source_keypoint),dim=1)
        target_keypoint_pred = self.modelA.sample(condition)
        condition = torch.cat((target_param,target_keypoint_pred),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_point_pred

    def forward(self,source_param,target_param,source_keypoint): 
        condition = torch.cat((source_param-target_param,source_keypoint),dim=1)
        target_keypoint_pred = self.modelA.sample(condition)
        condition = torch.cat((target_param,target_keypoint_pred),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_keypoint_pred,target_point_pred


class EK_PKVAE(nn.Module):
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB

    def editing_point(self,source_keypoint,target_keypoint,source_param): 
        condition = torch.cat((source_keypoint-target_keypoint,source_param),dim=1)
        target_param_pred = self.modelA.sample(condition)
        condition = torch.cat((target_param_pred,target_keypoint),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_point_pred

    def forward(self,source_keypoint,target_keypoint,source_param):  
        condition = torch.cat((source_keypoint-target_keypoint,source_param),dim=1)
        target_param_pred = self.modelA.sample(condition)
        condition = torch.cat((target_param_pred,target_keypoint),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_param_pred,target_point_pred