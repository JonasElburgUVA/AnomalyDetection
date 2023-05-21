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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="sample", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("--no_gpu", type=bool, default=False, help="Do not use gpu", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode
    no_gpu = args.no_gpu

    if no_gpu:
        device = torch.device('cpu')
    else:   
        device = torch.device('cuda')

    # DATASET SPECIFIC SETTINGS
    parameters = {"threshold_sample": 7,
                    "threshold_pixel_correct": 7,
                    "checkpoint_features": 'AnomalyDetection/src/DeepFake/checkpoints/ffhq_continued_020.pt',
                    "checkpoint_latent": 'AnomalyDetection/src/DeepFake/checkpoints/ffhq_ar_030.pt',
                    "vq_net":{"d":3,"n_channels":(16,32,64,256),"code_size":128,"n_res_block":2},
                    "ar_net":{"shape":(16,16), "n_block":4, "n_res_block":4, "n_channels":128}
    }
    
    vq_checkpoint = torch.load(parameters["checkpoint_features"], map_location=device)["model"]
    ar_checkpoint = torch.load(parameters["checkpoint_latent"], map_location=device)["model"]
    # INITIALIZE SMOOTHING OF PIXEL PREDICTIONS not sure if this is needed
    smooth = nn.Sequential(nn.MaxPool3d(kernel_size=3,padding=1,stride=1),
                            nn.AvgPool3d(kernel_size=(3,7,7),padding=(1,3,3),stride=1),
                        )    
    smooth.to(device)

    # LOAD MODEL
    vq_net = nets_LV.VQVAE(**parameters["vq_net"]).to(device)
    vq_net.eval()
    vq_net.load_state_dict(torch.load(parameters["checkpoint_features"], map_location=device)["model"])

    ar_net = nets_LV.VQLatentSNAIL(
        feature_extractor_model=vq_net,
        shape=(16, 16),
        n_block=4,
        n_res_block=4,
        n_channels=128
    ).to(device)
    ar_net.eval()
    ar_net.load_state_dict(torch.load(parameters["checkpoint_latent"], map_location=device)["model"])

    # LOAD DATA
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    eval_set = ImageFolder(input_dir, transform=transform)
    eval_loader = DataLoader(eval_set, batch_size=8, shuffle=False, num_workers=0)

    # PREDICT
    # if mode == "sample":
    with torch.no_grad():
        scores = []
        for i, (img, _) in tqdm(enumerate(eval_loader)):
            img.to(device)
            loss = ar_net.loss(img, reduction="none")["loss"].flatten(1)
            scores = torch.sum(loss * (loss > parameters["threshold_sample"]), 1).float()
            score = scores.sum() / 2000
            with open(os.path.join(output_dir,str(i)+ "score.txt"), "w") as write_file:
                write_file.write(str(score))





            



        

