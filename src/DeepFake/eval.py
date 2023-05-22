import os
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from zipfile import ZipFile
import numpy as np
import scipy.spatial.distance as distance
from collections import defaultdict

# def eval_dir(output_dir, label_dir, mode):
#     res = None
#     auroc = []
#     ap = []
#     if mode == "sample":
#         output_dir = os.path.join(ouptut_dir, "sample")
#         label_dir = os.path.join(label_dir, "sample")

import json


from scipy import stats

with open('data/output/FFHQ/scores.json') as s:
    scores = json.load(s)

res = defaultdict(lambda: {'auroc' : None, 'ap' : None})
for dif in scores.keys():
    
    # In the folders, 0 is anomalous and 1 is real
    labels = len(scores[dif]['0'].values())*[1] + len(scores[dif]['1'].values())*[0]
    pred = list(scores[dif]['0'].values()) + list(scores[dif]['1'].values())
    auroc = metrics.roc_auc_score(labels, pred)
    res[dif]['auroc'] = auroc

    ap = metrics.average_precision_score(labels, pred)
    res[dif]['ap'] = ap

    print("\n", dif)
    print(f"average precision: {ap}")
    print(f"area under roc curve: {auroc}")
    print(f"mean score anomalous: {np.mean(list(scores[dif]['0'].values()))}")
    print(f"mean score real: {np.mean(list(scores[dif]['1'].values()))}")
    t, p = stats.ttest_ind(list(scores[dif]['0'].values()), list(scores[dif]['1'].values()))
    print(f"t-test - t: {t}, p: {p}")

# print(res)
