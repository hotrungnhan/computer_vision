# https://gitlab.com/ronctli1012/blog1-pretrained-alexnet-and-visualization/-/blob/master/alexnet_main.py
# https://androidkt.com/feature-extraction-from-an-image-using-pre-trained-pytorch-model/
import os
import torch
from torch import nn, utils, optim
from torch.types import Device
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
import numpy as np

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image  # assumes that you're using GPU


image = image_loader("sample.jpg")
model = models.alexnet(pretrained=True)
model.load_state_dict(torch.load("alex.model"))
model.eval()
print(model(image))
