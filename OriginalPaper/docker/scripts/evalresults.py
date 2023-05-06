import nibabel as nib
import os
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from zipfile import ZipFile
import numpy as np
import scipy.spatial.distance as distance

def eval_dir(output_dir, label_dir, mode):
    res = 0
    AUROC = []
    averagePrecision = []
    if mode == "pixel":
        output_dir = os.path.join(output_dir, "pixel")
        label_dir = os.path.join(label_dir, "pixel")
        DSC = []

        for f in os.listdir(output_dir):
            vol = nib.load(os.path.join(output_dir, f))
            vol = np.asarray(vol.get_fdata())
            label = nib.load(os.path.join(label_dir, f))
            label = np.asarray(label.get_fdata())
            assert vol.shape == label.shape
            try:
                AUROC.append(metrics.roc_auc_score(label.flatten(),vol.flatten()))
                averagePrecision.append(metrics.average_precision_score(label.flatten(),vol.flatten()))
                DSC.append(1-distance.dice(label.flatten(),vol.flatten()))
            except:
                pass

        res = (np.mean(AUROC), np.mean(averagePrecision), np.mean(DSC))
        # AUROC, AP, and DSC
    elif mode == "sample":
        output_dir = os.path.join(output_dir, "sample")
        label_dir = os.path.join(label_dir, "sample")
        AUROC = []
        averagePrecision = []
        for f in os.listdir(output_dir):
            with open(os.path.join(output_dir, f), "r") as file:
                anomaly_score = file.read()
            with open(os.path.join(label_dir, f), "r") as file:
                label = file.read()
            # AUROC.append(metrics.roc_auc_score(anomaly_score, label))
            averagePrecision.append(metrics.average_precision_score(anomaly_score, label))
        res = (torch.mean(AUROC), torch.mean(averagePrecision))
            # AUROC and AP

    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory containing the predictions.",
        required=False,
        default="/Users/jonase/Documents/UVA/DEL2/AnomalyDetection/AnomalyDetection/OriginalPaper/output/full"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        help="Directory containing the ground truth labels.",
        required=False,
        default="/Users/jonase/Documents/UVA/DEL2/AnomalyDetection/AnomalyDetection/OriginalPaper/toy_label"
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Evaluation mode. Either 'pixel' or 'sample'.",
        required=False,
        default="sample"
    )
    args = parser.parse_args()

    res = eval_dir(args.output_dir, args.label_dir, args.mode)
    print(res)