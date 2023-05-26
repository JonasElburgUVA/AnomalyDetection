import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import nets_LV
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str, help="Path to data directory")
    parser.add_argument("--cp_dir", required=True, type=str, help="Path to checkpoint directory")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to output directory")
    parser.add_argument("-dataset", type=str, default="faceforensics", choices=["ffhq", "faceforensics"])
    parser.add_argument("-mode", type=str, default="sample",
                        help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("--no_gpu", type=bool, default=False,
                        help="Do not use gpu", required=False)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pruned", type=bool, default=False, help="Use pruned dataset for faceforensics")
    # parser.add_argument("--cutoff", type=int, default=15, help="Cutoff for pruned dataset")
    parser.add_argument("--ld", type=float, default=7, help="sample-wise threshold value")
    parser.add_argument("--thresholds", nargs='+',type=int, default=[5, 6, 7, 8, 9, 10], help="sample-wise threshold values for hyperparameter search")


    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output
    cp_dir = args.cp_dir
    dataset = args.dataset
    mode = args.mode
    no_gpu = args.no_gpu
    split = args.split
    pruned = args.pruned
    # cutoff = args.cutoff
    lambda_ = args.ld
    thresholds = args.thresholds
    txt_to_label = {'real': 0, 'fake': 1}
    difficulties = ['easy', 'medium', 'hard'] # only applicable to FFHQ
    if pruned:
        pr = "_pruned"
    else:
        pr = ""
    if split == "test":
        thresholds = [lambda_]
    else:
        thresholds = thresholds
    if no_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # DATASET SPECIFIC SETTINGS

    parameters = {"threshold_sample": lambda_,
                # "threshold_pixel_correct": 7,
                "checkpoint_features": os.path.join(cp_dir,f'{dataset}_vqvae.pt'),
                "checkpoint_latent": os.path.join(cp_dir,f'{dataset}_ar.pt'),
                "vq_net": {"d": 3, "n_channels": (16, 32, 64, 256), "code_size": 128, "n_res_block": 2},
                "ar_model": {"shape": (16, 16), "n_block": 4, "n_res_block": 4, "n_channels": 128}
                }

    vq_checkpoint = torch.load(
        parameters["checkpoint_features"], map_location=device)["model"]
    ar_checkpoint = torch.load(
        parameters["checkpoint_latent"], map_location=device)["model"]
    
    # LOAD MODEL
    vq_net = nets_LV.VQVAE(**parameters["vq_net"]).to(device)
    vq_net.eval()
    vq_net.load_state_dict(torch.load(
        parameters["checkpoint_features"], map_location=device)["model"])

    ar_model = nets_LV.VQLatentSNAIL(
        feature_extractor_model=vq_net,
        shape=(16, 16),
        n_block=4,
        n_res_block=4,
        n_channels=128
    ).to(device)
    ar_model.eval()
    ar_model.load_state_dict(torch.load(
        parameters["checkpoint_latent"], map_location=device)["model"])
    
    # INITIALIZE SMOOTHING OF PIXEL PREDICTIONS not sure if this is needed; only used for restoring the image
    smooth = nn.Sequential(nn.MaxPool3d(kernel_size=3, padding=1, stride=1),
                        nn.AvgPool3d(kernel_size=(3, 7, 7),
                                        padding=(1, 3, 3), stride=1),
                        )
    smooth.to(device)

    # LOAD DATA
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    if dataset == "ffhq":
        if split == "test":
            ffhq_sets = {}
            ffhq_loaders = {}
            for dif in difficulties:
                ffhq_sets[dif]= ImageFolder(os.path.join(data_dir, dataset, split, dif), transform=transform)
                ffhq_loaders[dif] = DataLoader(ffhq_sets[dif], batch_size=64, shuffle=False)
        elif split =="val":
            ffhq_set = ImageFolder(os.path.join(data_dir, dataset, split), transform=transform)
            ffhq_loader = DataLoader(ffhq_set, batch_size=64, shuffle=False)
    # For FaceForensics we load data differently, within the prediction loop, since we want to batch all images of the same video together
    # We do have to remove some folders that do not contain images, as a result of the preprocessing
    elif dataset == "faceforensics":
        for fr in ['real', 'fake']:
            n=0
            for cl in os.listdir(os.path.join(data_dir, dataset, f'{split}{pr}', fr)):
                # remove only empty dirs
                try: 
                    os.rmdir(os.path.join(data_dir, dataset, f'{split}{pr}', fr, cl)) 
                    n+=1
                except OSError: 
                    continue
            print(f"Removed {n} empty folders (videos) from {fr} set")


    # PREDICT
    print(f"Making predictions on {split} set of {dataset} with threshold(s) {thresholds}")
    output_dir = output_dir + f'/{dataset}'

    if dataset == "ffhq":
        predictions = {}
        for thr in thresholds:
            predictions[thr] = {}
            with torch.no_grad():
                if split == "test":
                    for dif in difficulties:
                        print(f"Predicting {dif} images with threshold {thr}")
                        predictions[thr][dif] = {'scores':[],'labels':[]}
                        for batch, cl in tqdm(ffhq_loaders[dif]):
                            loss = ar_model.loss(batch.to(device), reduction="none")["loss"].flatten(1)
                            scores = torch.sum(loss * (loss > thr), 1).float()
                            predictions[thr][dif]['scores'].append(scores.detach().cpu().numpy().tolist())
                            predictions[thr][dif]['labels'].append(cl.detach().cpu().numpy().tolist())
                elif split == "val":
                    print(f"Predicting validation images with threshold {thr}")
                    predictions[thr] = {'scores':[],'labels':[]}
                    for batch, cl in tqdm(ffhq_loader):
                        loss = ar_model.loss(batch.to(device), reduction="none")["loss"].flatten(1)
                        scores = torch.sum(loss * (loss > thr), 1).float()
                        predictions[thr]['scores'].append(scores.detach().cpu().numpy().tolist())
                        predictions[thr]['labels'].append(cl.detach().cpu().numpy().tolist())
                        # assert len(predictions[thr][dif]['scores']) == len(predictions[thr][dif]['labels'])
        with open(os.path.join(output_dir, f"scores_{split}.json"), "w") as write_file:
            json.dump(predictions, write_file)

    elif dataset == "faceforensics":
        input_dir = os.path.join(data_dir, dataset, f'{split}{pr}')
        predictions = defaultdict(lambda: defaultdict())
        for thr in thresholds:
            predictions[thr] = {'scores':[],'labels':[]}
            with torch.no_grad():
                for rf in os.listdir(input_dir):
                    for img_id in tqdm(os.listdir(os.path.join(input_dir, rf))):
                        imgdir = os.listdir(os.path.join(input_dir, rf, img_id)) 
                        batch = torch.stack([transform(Image.open(os.path.join(input_dir, rf, img_id, img))) for img in imgdir]).to(device)
                        loss = ar_model.loss(batch, reduction="none")["loss"].flatten(1)
                        scores = torch.sum(loss * (loss > thr), 1).float()
                        score = scores.mean()
                        predictions[thr]['scores'].append(score.detach().cpu().numpy().tolist())
                        predictions[thr]['labels'].append(txt_to_label[rf])
        with open(os.path.join(output_dir,f"scores_{split}.json"), "w") as write_file:
            json.dump(predictions, write_file)