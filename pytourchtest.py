import os
import torch
from torch import nn, utils, optim
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
#
model = models.alexnet(pretrained=True)
lastfclayer = model.classifier[-1]
print(lastfclayer)
model2 = models.resnet18(pretrained=True)
print(model2.fc.in_features)
