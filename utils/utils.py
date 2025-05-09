import wandb
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from models.models import *
from dataloaders import *


def make(config, device="cuda"):
    train_loader = make_comics_loader(split="train",
                                      batch_size=config.batch_size,
                                      max_samples=config.max_samples,
                                      shuffle=True)
    val_loader = make_comics_loader(split="val",
                                    batch_size=config.batch_size,
                                    max_samples=config.max_samples,
                                    shuffle=False)
    test_loader = make_comics_loader(split="test",
                                     batch_size=config.batch_size,
                                     max_samples=config.max_samples,
                                     shuffle=False)

    model = Baseline(
        embedding_dim=config.embedding_dim,
        freeze_bert=config.freeze_bert,
        freeze_resnet=config.freeze_resnet,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer

