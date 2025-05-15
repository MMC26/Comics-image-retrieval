import wandb
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from models.models import *
from dataloaders import *


def make(config, device="cuda"):
    print("Carregant train_loader...")
    train_loader = make_comics_loader(split="train",
                                      batch_size=config.batch_size,
                                      max_samples=config.max_samples,
                                      shuffle=True,
                                      num_workers=config.num_workers)
    print("Carregant val_loader...")
    val_loader = make_comics_loader(split="val",
                                    batch_size=config.batch_size,
                                    max_samples=config.max_samples,
                                    shuffle=False,
                                    num_workers=config.num_workers)
    print("Carregant test_loader...")
    test_loader = make_comics_loader(split="test",
                                     batch_size=config.batch_size,
                                     max_samples=config.max_samples,
                                     shuffle=False,
                                      num_workers=config.num_workers)

    model = Baseline(
        embedding_dim=config.embedding_dim,
        freeze_bert=config.freeze_bert,
        freeze_resnet=config.freeze_resnet,
    ).to(device)

    criterion = nn.TripletMarginLoss(margin=0.2, p=2, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer

