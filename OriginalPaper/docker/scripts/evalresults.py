import pred_simple
import os
import torch
import torch.nn.functional as F

def eval_dir(output_dir, labels, mode):
    res = 0
    if mode == "pixel":
        for f in os.listdir(output_dir):
            # AUROC, AP, and DSC
    elif mode == "sample":
        for f in os.listdir(output_dir):
            # AUROC and AP

    return res

def auroc():
    return auroc

def averagePrecision():
    return ap

def dsc():
    return dsc