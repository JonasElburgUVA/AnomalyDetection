import os
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from zipfile import ZipFile
import numpy as np
import scipy.spatial.distance as distance

def eval_dir(output_dir, label_dir, mode):
    res = None
    auroc = []
    ap = []
    if mode == "sample":
        output_dir = os.path.join(ouptut_dir, "sample")
        label_dir = os.path.join(label_dir, "sample")

