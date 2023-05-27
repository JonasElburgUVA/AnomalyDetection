import torch
import utils
import nets_LV
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse


parser = argparse.ArgumentParser(
    prog="Training AR modek",
    description="This program runs the training for the AR model"
)

parser.add_argument("--training_directory", type=str,
                    help="The directory with the training images.")
parser.add_argument("--holdout_directory", type=str,
                    help="The directory where the holdout set is located for training.")
parser.add_argument("--vqvae_checkpoint", type=str, required=True,
                    help="When present, we will continue training, otherwise train from scratch.")
parser.add_argument("--ar_checkpoint", type=str, default=None,
                    help="When present, we will continue training, otherwise train from scratch.")
parser.add_argument('--epochs', type=int, default=30,
                    help="Number of epochs to train for.")

args = parser.parse_args()


device = torch.device("cuda")

transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.3, hue=.2),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor()
])

train_dir = args.training_directory
holdout_dir = args.holdout_directory

train_dataset = ImageFolder(train_dir, transform=transform_pipeline)
val_dataset = ImageFolder(holdout_dir, transform=transform_pipeline)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)


vq_model = nets_LV.VQVAE(
    d=3,
    n_channels=(16, 32, 64, 256),
    code_size=128,
    n_res_block=2,
    dropout_p=.1
)

vq_checkpoint = torch.load(args.vqvae_checkpoint)
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
    ar_model.load_state_dict(ar_checkpoint['model'])
    optimizer.load_state_dict(ar_checkpoint['optimizer'])


wandb_config = {
    "architecture": "AR",
    "dataset": "FFHQ",
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
    chpt="ffhq_ar",
    log_recon_metrics=False
)
