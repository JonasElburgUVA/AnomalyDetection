import torch
import utils
import nets_LV
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


device = torch.device("cuda")

transform_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.3, hue=.2),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor()
])

train_dataset = ImageFolder("images/ffhq/train/", transform=transform_pipeline)
val_dataset = ImageFolder("images/ffhq/validation/", transform=transform_pipeline)

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

utils.train_epochs(
    model=model,
    optimizer=optimizer,
    tracker=tracker,
    train_loader=train_dataloader,
    test_loader=val_dataloader,
    epochs=30,
    device=device,
    config=None,
    chpt="ffhq"
)