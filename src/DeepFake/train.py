import torch
import utils
import nets_LV
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse

parser = argparse.ArgumentParser(
    prog="Training VQ-VAE model",
    description="This program runs the training for the VQ-VAE model"
)

parser.add_argument("--training_directory", type=str,
                    help="The directory with the training images.")
parser.add_argument("--holdout_directory", type=str,
                    help="The directory where the holdout set is located for training.")
parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                    help="When present, we will continue training, otherwise train from scratch.")
parser.add_argument('--epochs', type=int, default=30,
                    help="Number of epochs to train for.")

args = parser.parse_args()

device = torch.device("cuda")

train_dir = args.training_directory
holdout_dir = args.holdout_directory

transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.3, hue=.2),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(train_dir, transform=transform_pipeline)
val_dataset = ImageFolder(holdout_dir, transform=transform_pipeline)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

model = nets_LV.VQVAE(
    d=3,
    n_channels=(16, 32, 64, 256),
    code_size=128,
    n_res_block=2,
    dropout_p=.1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
tracker = utils.train_tracker()

if args.vqvae_checkpoint is not None:
    checkpoint = torch.load(args.vqvae_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

wandb_config = {
    "architecture": "VQ-VAE",
    "dataset": "FFHQ",
    "epochs": args.epochs,
}

utils.train_epochs(
    model=model,
    optimizer=optimizer,
    tracker=tracker,
    train_loader=train_dataloader,
    test_loader=val_dataloader,
    epochs=args.epochs,
    device=device,
    config=wandb_config,
    chpt="ffhq",
    log_recon_metrics=True
)
