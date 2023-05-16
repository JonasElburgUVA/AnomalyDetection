import torch
import wandb

import matplotlib.pyplot as plt
import os
from tqdm import tqdm


wandb.login()


def train(model, train_loader, optimizer, device):
    model.train()

    train_losses = []
    batch_sizes = []
    train_ssim, train_psnr = [], []

    for img, _ in tqdm(train_loader):
        loss = model.loss(img.to(device))

        optimizer.zero_grad()
        loss["loss"].backward()

        train_ssim.append(loss["ssim"].item() * img.shape[0])
        train_psnr.append(loss["psnr"].item() * img.shape[0])
        train_losses.append(loss["loss"].item() * img.shape[0])

        # Gradient clippling
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)

        optimizer.step()

        batch_sizes.append(img.shape[0])

    train_loss = sum(train_losses) / sum(batch_sizes)
    train_ssim = sum(train_ssim) / sum(batch_sizes)
    train_psnr = sum(train_psnr) / sum(batch_sizes)

    return train_loss, train_ssim, train_psnr


def eval_loss(model, data_loader, device):
    model.eval()

    eval_losses = []
    batch_sizes = []
    val_ssim, val_psnr = [], []

    with torch.no_grad():
        for img, _ in data_loader:
            loss = model.loss(img.to(device))

            val_ssim.append(loss["ssim"].item() * img.shape[0])
            val_psnr.append(loss["psnr"].item() * img.shape[0])
            eval_losses.append(loss['loss'].item() * img.shape[0])

            batch_sizes.append(img.shape[0])

    eval_loss = sum(eval_losses) / sum(batch_sizes)
    val_ssim = sum(val_ssim) / sum(batch_sizes)
    val_psnr = sum(val_psnr) / sum(batch_sizes)

    return eval_loss, val_ssim, val_psnr


def save_checkpoint(model, optimizer, tracker, file_name):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'tracker': tracker,
    }

    torch.save(checkpoint, file_name)


class train_tracker:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.lr = []

    def __len__(self):
        return len(self.train_losses)

    def append(self, train_loss, test_loss, lr):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.lr.append(lr)

    def plot(self, N=None):
        N = N if N is not None else self.__len__()
        plt.plot(self.train_losses[-N:], label='Train')
        plt.plot(self.test_losses[-N:], label='Eval')
        plt.legend()
        plt.show()


def train_epochs(model, optimizer, tracker, train_loader, test_loader, epochs, device, config, chpt=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project="AnomalyDetection",
        # track hyperparameters and run metadata
        config=config
    )

    print('Starting training')

    for epoch in range(epochs):
        train_loss, train_ssim, train_psnr = train(model, train_loader, optimizer, device)
        test_loss, val_ssim, val_psnr = eval_loss(model, test_loader, device)

        wandb.log({'Loss/train': train_loss})
        wandb.log({'Loss/test': test_loss})

        wandb.log({'Metrics/train_ssim': train_ssim})
        wandb.log({'Metrics/train_psnr': train_psnr})

        wandb.log({'Metrics/val_ssim': val_ssim})
        wandb.log({'Metrics/val_psnr': val_psnr})

        tracker.append(train_loss, test_loss, optimizer.param_groups[0]['lr'])

        print('{} epochs, {:.3f} test loss, {:.3f} train loss'.format(
            len(tracker), test_loss, train_loss))
        if chpt is not None:
            if (not os.path.isdir("checkpoints")):
                print('Created folder "checkpoints".')
                os.mkdir('checkpoints')
                
            save_checkpoint(model, optimizer, tracker,
                            'checkpoints/{}_{:03}.pt'.format(chpt, len(tracker)))

    wandb.finish()


def dice_score(trues, preds):
    """Calculate dice score / f1 given binary boolean variables: 2 x IoU"""
    return 2. * (trues & preds).sum()/(trues.sum() + preds.sum())


def max_score(trues, pred, score_func=dice_score, steps=8):
    """Iterate through possible threshold ranges and return max score and argmax threshold """
    min_d, max_d = pred.min(), pred.max()

    for i in range(steps):
        mid_d = (max_d-min_d)/2 + min_d
        mid_s = score_func(trues, pred > mid_d)

        q1_s = score_func(trues, pred > (max_d-min_d)/4 + min_d)
        q3_s = score_func(trues, pred > 3*(max_d-min_d)/4 + min_d)

        if q1_s == q3_s:
            break
        elif q1_s > q3_s:
            max_d = mid_d
        else:
            min_d = mid_d
    return mid_s, mid_d
