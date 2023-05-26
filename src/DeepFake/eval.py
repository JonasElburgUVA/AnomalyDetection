import os
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from zipfile import ZipFile
import numpy as np
import scipy.spatial.distance as distance
from collections import defaultdict
import json
from scipy import stats
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, help="Path to prediction directory, containing predictions in JSON format")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--dataset", type=str, default="faceforensics", choices=["ffhq", "faceforensics"])
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    output_dir = args.output_dir
    dataset = args.dataset
    split = args.split

    with open(os.path.join(pred_dir, f"{dataset}", f"scores_{split}.json")) as s:
        scores = json.load(s)

    res = defaultdict(lambda: defaultdict(lambda: {'auroc': None, 'ap': None}))
    thresholds = list(scores.keys())
    print(f"threshods: {thresholds}")
    if dataset == 'ffhq':
        for thr in thresholds:
            thr = str(thr)
            for dif in ['easy', 'medium', 'hard']:
                labels = np.concatenate(scores[thr][dif]['labels'])
                preds = np.concatenate(scores[thr][dif]['scores'])

                auroc = metrics.roc_auc_score(labels,preds)
                ap = metrics.average_precision_score(labels,preds, pos_label=1)
                print(f"threshold: {thr}, difficulty: {dif}, auroc: {auroc}, ap: {ap}")

            res[thr][dif]['auroc'] = auroc
            res[thr][dif]['ap'] = ap

    elif dataset == 'faceforensics':
        res = defaultdict(lambda: {'auroc' : None, 'ap': None})
        for thr in thresholds:
            thr = str(thr)
            labels = scores[thr]['labels']
            preds = scores[thr]['scores']

            auroc = metrics.roc_auc_score(labels,preds)
            ap = metrics.average_precision_score(labels,preds, pos_label=1)

            res[thr]['auroc'] = auroc
            res[thr]['ap'] = ap
    print(res.items())
        # for dif in scores.keys():
        #     # In the folders, 0 is anomalous and 1 is real
        #     labels = len(scores[dif]['0'].values())*[1] + \
        #         len(scores[dif]['1'].values())*[0]
                
        #     pred = list(scores[dif]['0'].values()) + list(scores[dif]['1'].values())
        #     auroc = metrics.roc_auc_score(labels, pred)
        #     res[dif]['auroc'] = auroc

        #     ap = metrics.average_precision_score(labels, pred)
        #     res[dif]['ap'] = ap

        #     print("\n", dif)
        #     print(f"average precision: {ap}")
        #     print(f"area under roc curve: {auroc}")
        #     print(f"mean score anomalous: {np.mean(list(scores[dif]['0'].values()))}")
        #     print(f"mean score real: {np.mean(list(scores[dif]['1'].values()))}")
        #     t, p = stats.ttest_ind(
        #         list(scores[dif]['0'].values()), list(scores[dif]['1'].values()))
        #     print(f"t-test - t: {t}, p: {p}")

        # with open(os.path.join(scores_dir, "metrics.json"), "w") as write_file:
        #     json.dump(res, write_file)

        # # print(res)
