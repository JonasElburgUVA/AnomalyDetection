import torch
import utils
import nets_LV
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from faceforensics_dataloader import FaceForensicsDataset, collate_fn


device = torch.device("cuda")

data_dir = "/project/gpuuva022/shared/face_forensics_extracted/original_faces"

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
    data_dir,
    split="train",
    sample_number=8,
    transform=train_transform_pipeline
)

val_dataset = FaceForensicsDataset(
    data_dir,
    split="test",
    sample_number=16, # 2240 validation samples
    transform=val_transform_pipeline
)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

vq_checkpoint = torch.load(
    "/home/lcur1720/anomaly-latest/AnomalyDetection/checkpoints_vqvae/faceforensics_200.pt"
)

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

wandb_config = {
    "architecture": "AR",
    "dataset": "FaceForensics",
    "epochs": 200,
}

utils.train_epochs(
    model=ar_model,
    optimizer=optimizer,
    tracker=tracker,
    train_loader=train_dataloader,
    test_loader=val_dataloader,
    epochs=200,
    device=device,
    config=wandb_config,
    chpt="faceforensics_ar",
    log_recon_metrics=False
)
