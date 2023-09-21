# -*- codeing=utf-8 -*—
# @time: 2023/5/18 21:02
# @Author : 李明聪
# @File ： flop.py
# @software : PyCharm
import torch
from thop import profile

from rail_marking.segmentation.models import BiSeNetV2

net = BiSeNetV2(n_classes=21)  # 定义好的网络模型

from torchstat import stat
import torchvision.models as models

model = net
stat(model, (3, 512, 512))