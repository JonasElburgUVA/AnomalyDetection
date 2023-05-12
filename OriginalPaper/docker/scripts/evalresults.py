import nibabel as nib
import os
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from zipfile import ZipFile
import numpy as np
import scipy.spatial.distance as distance

def get_dsc(label, vol, t=0.35):
    # threshold t is a hyperparameter
    tp = sum(vol[label==1] >= t)
    fp = sum(vol[label==0] >= t)
    fn = sum(vol[label==1] < t)

    dsc= 2 * tp / (2 * tp + fp + fn)
    if dsc != dsc:
        raise ValueError("NaN encountered in DSC calculation.")
    return dsc

def eval_dir(output_dir, label_dir, mode):
    res = None
    auroc = []
    ap = []
    if mode == "pixel":
        output_dir = os.path.join(output_dir, "pixel")
        label_dir = os.path.join(label_dir, "pixel")
        dsc = []

        for f in os.listdir(output_dir):
            vol = nib.load(os.path.join(output_dir, f))
            vol = np.asarray(vol.get_fdata()).flatten()
            label = nib.load(os.path.join(label_dir, f))
            label = np.asarray(label.get_fdata()).flatten()
            assert vol.shape == label.shape
            try:
                auroc.append(metrics.roc_auc_score(label,vol))
                ap.append(metrics.average_precision_score(label, vol))
                dsc.append(get_dsc(label, vol))
            except:
                print(f"Could not calculate metrics for {f}. Does the label contain only 0s or 1s? Skipping...")
                pass
        res = (np.mean(auroc), np.mean(ap), np.mean(dsc))
        
    elif mode == "sample":
        output_dir = os.path.join(output_dir, "sample")
        label_dir = os.path.join(label_dir, "sample")
        auroc = []
        ap = []
        for f in os.listdir(output_dir):
            with open(os.path.join(output_dir, f), "r") as file:
                anomaly_score = file.read()
            with open(os.path.join(label_dir, f), "r") as file:
                label = file.read()
            auroc.append(metrics.roc_auc_score(anomaly_score, label))
            ap.append(metrics.average_precision_score(anomaly_score, label))
        res = (torch.mean(auroc), torch.mean(ap))
    else:
        raise ValueError("Invalid mode. Must be either 'pixel' or 'sample'.")

    return res

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory containing the predictions.",
        required=True,
        default="../../output/full"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        help="Directory containing the ground truth labels.",
        required=True,
        default="../../toy_label"
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Evaluation mode. Either 'pixel' or 'sample'.",
        required=False,
        default="pixel"
    )
    args = parser.parse_args()

    

    res = dict(zip(['AUROC', 'AP', 'DSC'], eval_dir(args.output_dir, args.label_dir, args.mode)))
    print(res)


    # def getaverageprecision(labels, scores):
#     thresholds = np.arange(0, 1, 0.01)
#     precision = np.zeros_like(thresholds)
#     recall = np.zeros_like(thresholds)
#     if sum(labels==1) == 0:
#         return 1.0
#     else:
#         for i, t in enumerate(thresholds):
#             precision[i] = sum((scores >= t) & (labels==1)) / sum(scores >= t)
#             recall[i] = sum((scores >= t) & (labels==1)) / sum(labels==1)

#     AP = 0
#     for i in range(1, len(thresholds)-1):
#         AP += precision[i] * (recall[i] - recall[i-1])

#     return AP

# def getauroc(labels, scores):
#     thresholds = np.arange(0, 1, 0.01)
#     tpr = np.zeros_like(thresholds)
#     fpr = np.zeros_like(thresholds)
#     for i, t in enumerate(thresholds):
#         if len(scores[labels==1]) > 0:
#             tpr[i] = np.mean(scores[labels == 1] >= t)
#         else:
#             tpr[i] = 1.0
#         fpr[i] = np.mean(scores[labels == 0] >= t)
#     return metrics.auc(fpr, tpr)