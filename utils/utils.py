import wandb
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from models.models import *
from dataloaders import *


class CosineTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = (anchor * positive).sum(dim=1)
        neg_sim = (anchor * negative).sum(dim=1)
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0)
        return loss.mean()


def make(config, device="cuda"):
    print("Carregant train_loader...")
    train_loader = make_comics_loader(split="train",
                                      batch_size=config.batch_size,
                                      max_samples=config.max_samples,
                                      shuffle=True,
                                      preprocessed=False,
                                      num_workers=config.num_workers)
    print("Carregant val_loader...")
    val_loader = make_comics_loader(split="val",
                                    batch_size=config.batch_size,
                                    max_samples=config.max_samples,
                                    shuffle=False,
                                    preprocessed=False,
                                    num_workers=config.num_workers)
    print("Carregant test_loader...")
    test_loader = make_comics_loader(split="test",
                                     batch_size=config.batch_size,
                                     max_samples=config.max_samples,
                                     shuffle=False,
                                     preprocessed=False,
                                     num_workers=config.num_workers)

    model = Baseline(
        embedding_dim=config.embedding_dim,
        freeze_bert=config.freeze_bert,
        freeze_resnet=config.freeze_resnet,
        use_sentence_bert=config.get("use_sentence_bert", True),
        use_gtr=config.get("use_gtr", False),
        gtr_model_name=config.get("gtr_model_name", "sentence-transformers/gtr-t5-base"),
        fine_tune_resnet_tail=False,
        remove_first_sentence=config.get("remove_first_sentence", False)
    ).to(device)

    criterion = CosineTripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, val_loader, test_loader, criterion, optimizer

