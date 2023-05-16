import torch
import utils
import nets_LV
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

train_dir = '../../../../project/gpuuva022/shared/AnomalyDetection/FFHQ_Data/FFHQ_data/train_ffhq/'
holdout_dir = '../../../../project/gpuuva022/shared/AnomalyDetection/FFHQ_Data/FFHQ_data/holdout_ffhq'
val_dir = '../../../../project/gpuuva022/shared/AnomalyDetection/FFHQ_Data/FFHQ_data/val_ffhq'

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

device = torch.device('cuda')
feat_ext_mdl = nets_LV.VQVAE(
    d=3,
    n_channels=(16, 32, 64, 256),
    code_size=128,
    n_res_block=2,
    dropout_p=.1
)

model = nets_LV.VQLatentSNAIL(feature_extractor_model=feat_ext_mdl,
                              shape=(20, 20), n_block=4, n_res_block=4, n_channels=128)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
tracker = utils.train_tracker()

wandb_config = {
    "architecture": "AR",
    "dataset": "FFHQ",
    "epochs": 30,
}

utils.train_epochs(
    model=model,
    optimizer=optimizer,
    tracker=tracker,
    train_loader=train_dataloader,
    test_loader=val_dataloader,
    epochs=40,
    device=device,
    config=wandb_config,
    chpt='AR')
