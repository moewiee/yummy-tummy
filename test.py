import torch
from cvcore.configs import get_cfg_defaults
from cvcore.modeling.semantic_seg import UNetSegmentor, FPNSegmentor

x = torch.randn(2, 6, 256, 256)
cfg = get_cfg_defaults()
# cfg.merge_from_file("cvcore/configs/ResNet50UNet.yaml")
cfg.merge_from_file('cvcore/configs/B3FPN.yaml')
# model = UNetSegmentor(cfg)
model = FPNSegmentor(cfg)
print(model.fpn)
# print(model.backbone._out_feature_strides)
# print(model.sem_seg_head)
# print(model)
# print(model(x).shape)
# model(x)