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
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-dataset", type=str, default="faceforensics", choices=["ffhq", "faceforensics"])
    parser.add_argument("-mode", type=str, default="sample",
                        help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("--no_gpu", type=bool, default=False,
                        help="Do not use gpu", required=False)

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output
    dataset = args.dataset
    mode = args.mode
    no_gpu = args.no_gpu

    if no_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')


    # DATASET SPECIFIC SETTINGS
    if dataset == "ffhq":
        parameters = {"threshold_sample": 7,
                    "threshold_pixel_correct": 7,
                    "checkpoint_features": 'AnomalyDetection/src/DeepFake/checkpoints/ffhq_continued_020.pt',
                    "checkpoint_latent": 'AnomalyDetection/src/DeepFake/checkpoints/ffhq_ar_030.pt',
                    "vq_net": {"d": 3, "n_channels": (16, 32, 64, 256), "code_size": 128, "n_res_block": 2},
                    "ar_net": {"shape": (16, 16), "n_block": 4, "n_res_block": 4, "n_channels": 128}
                    }

    elif dataset == "faceforensics":
        parameters = {"threshold_sample": 7,
                    "threshold_pixel_correct": 7,
                    "checkpoint_features": 'src/DeepFake/checkpoints/faceforensics_vqvae_200.pt',
                    "checkpoint_latent": 'src/DeepFake/checkpoints/faceforensics_ar_200.pt',
                    "vq_net": {"d": 3, "n_channels": (16, 32, 64, 256), "code_size": 128, "n_res_block": 2},
                    "ar_net": {"shape": (16, 16), "n_block": 4, "n_res_block": 4, "n_channels": 128}
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

    ar_net = nets_LV.VQLatentSNAIL(
        feature_extractor_model=vq_net,
        shape=(16, 16),
        n_block=4,
        n_res_block=4,
        n_channels=128
    ).to(device)
    ar_net.eval()
    ar_net.load_state_dict(torch.load(
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

    # PREDICT
    # if mode == "sample":
    if dataset == "ffhq":
        with torch.no_grad():
            predictions = defaultdict()

            for difficulty in os.listdir(data_dir):
                for real_or_fake in os.listdir(os.path.join(data_dir, difficulty)):
                    for file_id in tqdm(os.listdir(os.path.join(data_dir, difficulty, real_or_fake))):
                        img = Image.open(os.path.join(data_dir, difficulty, real_or_fake, file_id))
                        img = torch.unsqueeze(transform(img).to(device), 0)
                        loss = ar_net.loss(img, reduction="none")["loss"].flatten(1)
                        scores = torch.sum(loss * (loss > parameters["threshold_sample"]), 1).float()
                        score = scores.sum()
                        predictions[difficulty][real_or_fake][file_id] = score.detach().cpu().numpy().tolist()

            with open(os.path.join(output_dir, "ffhq", "scores.json"), "w") as write_file:
                json.dump(predictions, write_file)

    elif dataset == "faceforensics":
        if mode == "sample":
            with torch.no_grad():
                predictions = defaultdict(lambda: defaultdict(dict))
                for real_or_fake in os.listdir(data_dir):
                    for vid in tqdm(os.listdir(os.path.join(data_dir, real_or_fake))):
                        for file_id in os.listdir(os.path.join(data_dir, real_or_fake, vid)):
                            img = Image.open(os.path.join(data_dir, real_or_fake, vid, file_id))
                            img = torch.unsqueeze(transform(img).to(device), 0)
                            loss = ar_net.loss(img, reduction="none")["loss"].flatten(1)
                            scores = torch.sum(loss * (loss > parameters["threshold_sample"]), 1).float()
                            score = scores.sum()
                            predictions[real_or_fake][vid][file_id] = score.detach().cpu().numpy().tolist()
                with open(os.path.join(output_dir, "faceforensics","scores_sample.json"), "w") as write_file:
                    json.dump(predictions, write_file)

        elif mode == "pixel":
            with torch.no_grad():
                predictions = {}
                for real_or_fake in os.listdir(data_dir):
                    for vid in os.listdir(os.path.join(data_dir, real_or_fake)):
                        for file_id in tqdm(os.listdir(os.path.join(data_dir, real_or_fake, vid))):
                            img = Image.open(os.path.join(data_dir, real_or_fake, vid, file_id))
                            img = torch.unsqueeze(transform(img).to(device), 0)
                            x_tilde, z_tilde = ar_net.reconstruct(15, img, threshold_log_p = parameters["threshold_pixel_correct"])
                            x_tilde = -smooth(-x_tilde.unsqueeze(0).unsqueeze(0)).squeeze()
                            predictions[real_or_fake][vid][file_id] = x_tilde.detach().cpu().numpy().tolist()
                with open(os.path.join(output_dir, "faceforensics","scores_pixel.json"), "w") as write_file:
                    json.dump(predictions, write_file)


    # elif mode == "pixel":
    #     with torch.no_grad():
    #      for img in tqdm(os.listdir(input_dir)):
    #         img = Image.open(os.path.join(input_dir, img))
    #         img = torch.unsqueeze(transform(img).to(device), 0)
    #         x_tilde, z_tilde = reconstruct(15, img, threshold_log_p = parameters["threshold_pixel_correct"])

#             x_tilde = -smooth(-x_tilde.unsqueeze(0).unsqueeze(0)).squeeze()

    # with torch.no_grad():
    #     scores = []
    #     for i, (img, _) in tqdm(enumerate(eval_loader)):
    #         img.to(device)
    #         loss = ar_net.loss(img, reduction="none")["loss"].flatten(1)
    #         scores = torch.sum(loss * (loss > parameters["threshold_sample"]), 1).float()
    #         score = scores.sum() / 2000
    #         with open(os.path.join(output_dir,str(i)+ "score.txt"), "w") as write_file:
    #             write_file.write(str(score))
