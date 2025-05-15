import torch
import wandb
import random
import numpy as np

from test import qualitative_test
from train import *
from utils.utils import *
from dataloaders import *

# Semilles per reproductibilitat
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_pipeline(cfg: dict):
    # tell wandb to get started
    with wandb.init(project=cfg["project_name"], config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config, device=device)

        # and use them to train the model
        train(model, train_loader, val_loader, optimizer,criterion, config, device=device)

        # and validate
        # validate(model, val_loader, device=device)

        qualitative_test(model, test_loader, device=device)

        return model


if __name__ == "__main__":
    wandb.login()

    config = dict(
        embedding_dim=512,
        freeze_bert=True,
        freeze_resnet=True,
        learning_rate=1e-4,
        batch_size=32,
        max_samples=None,
        epochs=5,
        project_name="comics-retrieval",
        checkpoint_path="checkpoint.pt",
        num_workers=0
    )

    model = model_pipeline(config)