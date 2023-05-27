import torch
import utils
import nets_LV
import argparse
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from faceforensics_dataloader import FaceForensicsDataset, collate_fn


parser = argparse.ArgumentParser(
    prog="Training VQ-VAE model (FaceForensics)",
    description="This program runs the training for the VQ-VAE model on FaceForensics"
)

parser.add_argument(
    "--data_directory",
    type=str,
    help="The dataset directory, containing train and test folders"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="Number of epochs to train for."
)

parser.add_argument(
    "--vqvae_checkpoint",
    type=str,
    required=True,
    help="The checkpoint for the VQ-VAE model"
)

parser.add_argument(
    "--ar_checkpoint",
    type=str,
    default=None,
    help="When present, we will continue training, otherwise train from scratch."
)

device = torch.device("cuda")
args = parser.parse_args()

train_transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.3, hue=.2),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor()
])

val_transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = FaceForensicsDataset(
    args.data_directory,
    split="train",
    sample_number=8,
    transform=train_transform_pipeline
)

val_dataset = FaceForensicsDataset(
    args.data_directory,
    split="test",
    sample_number=16, # 2240 validation samples
    transform=val_transform_pipeline
)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

vq_checkpoint = torch.load(args.vqvae_checkpoint)

vq_model = nets_LV.VQVAE(
    d=3,
    n_channels=(16, 32, 64, 256),
    code_size=128,
    n_res_block=2,
    dropout_p=.1
)

vq_model.load_state_dict(vq_checkpoint["model"])
vq_model = vq_model.to(device)

ar_model = nets_LV.VQLatentSNAIL(
    feature_extractor_model=vq_model,
    shape=(16, 16),
    n_block=4,
    n_res_block=4,
    n_channels=128
).to(device)

optimizer = optim.Adam(ar_model.parameters(), lr=1e-4)
tracker = utils.train_tracker()

if args.ar_checkpoint is not None:
    ar_checkpoint = torch.load(args.ar_checkpoint)

    ar_model.load_state_dict(ar_checkpoint["model"])
    optimizer.load_state_dict(ar_checkpoint["optimizer"])
    tracker.load_state_dict(ar_checkpoint["tracker"])

wandb_config = {
    "architecture": "AR",
    "dataset": "FaceForensics",
    "epochs": args.epochs,
}

utils.train_epochs(
    model=ar_model,
    optimizer=optimizer,
    tracker=tracker,
    train_loader=train_dataloader,
    test_loader=val_dataloader,
    epochs=args.epochs,
    device=device,
    config=wandb_config,
    chpt="faceforensics_ar",
    log_recon_metrics=False
)
