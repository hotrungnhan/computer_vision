from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch
from kornia.feature import SIFTDescriptor
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import splitfolders
import random
import os
from tqdm import tqdm
from sklearn import svm

a1 = np.empty((0, 128))
a2 = np.empty((23, 128))
a4 = np.empty(0)
a3 = np.array([5, 8, 12])
print(np.vstack(a1, a2).shape)
print(np.concatenate((a3, a4)).shape)
