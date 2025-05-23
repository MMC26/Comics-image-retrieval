import torch
import wandb
import random
import numpy as np

from test import *
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

texts = [
    "The central figure is a man with distinctive features that immediately draw the eye. He sports a green suit jacket, a white shirt, and a black tie, giving him a sharp, professional appearance. His most striking characteristic is his vibrant red hair, which is complemented by long sideburns that add to his dramatic look. Topping off his ensemble is a blue bowler hat, a classic accessory often associated with the era's detective and criminal archetypes. The man is captured in a dynamic pose, leaning over what appears to be a wooden desk or counter. His body language suggests he's in",
    "The setting is a dark, foreboding forest with a large, menacing tank approaching from the right side of the frame. In the foreground, we see three main characters: 1. A man in a green suit, running frantically to the left. He's wearing a hat and appears to be in a state of panic. 2. A woman in a brown trench coat and blue boots, also fleeing to the left. Her long hair flows behind her as she runs. 3. Another man, dressed in a blue suit and hat, running in the opposite direction to the right. His actions suggest he's",
    "On the left side of the panel, we see a muscular black man wearing a blue turban. He's shirtless and has a fierce expression on his face. The man is pointing his finger upwards, seemingly in the middle of giving an order or scolding someone. On the right side of the panel, there's a woman who appears to be the recipient of the man's rebuke. She has blonde hair and is wearing a green dress with a yellow belt. What's particularly striking about her appearance is the red cape draped over her shoulders, giving her a heroic or regal air.",
    " The focal point is a character's foot, clad in a brown boot with a distinctive sole pattern featuring small circles. The character is wearing orange pants, which add a vibrant splash of color to the scene. The character is in the process of stepping on a small, cylindrical object on the ground. This object appears to be a capsule or pill, and its presence is emphasized by a white, spiky outline surrounding it, suggesting it's about to be stepped on."
]


def model_pipeline(cfg: dict):
    # tell wandb to get started
    with wandb.init(project=cfg["project_name"], config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config, device=device)

        # and use them to train the model
        train(model, train_loader, val_loader, optimizer, criterion, config, device=device)

        # and validate
        validate(model, val_loader, criterion, device=device)

        # qualitative_test(model, test_loader, device=device)
        # visualize_top10_live(model, test_loader, device=device,num_texts=3)

        # prova amb uns textos fixos, els podem variar o no provar, era per veure com variava segons els model
        # visualize_fixed_texts(model, test_loader, texts, device=device)
        visualize_fixed_texts_from_loader(model, test_loader, device=device, indices=[5, 12, 20])
        analyze_global_validation_similarities(model, val_loader, device="cuda", fixed_indices=[2, 20, 200])
        plot_global_recall_distribution(model, val_loader, device=device)

        return model


if __name__ == "__main__":
    wandb.login()

    config = dict(
        embedding_dim=768,  # 384 amb Sentence-bert i 768 amb GTR
        freeze_bert=True,
        freeze_resnet=True,
        learning_rate=1e-4,
        batch_size=1,
        max_samples=1,
        epochs=1,
        project_name="comics-retrieval",
        checkpoint_path="checkpoint_TrueBert.pt",
        num_workers=0,
        use_sentence_bert=False,
        fine_tune_resnet_tail=False,
        remove_first_sentence=True,
        use_gtr=True,

    )

    model = model_pipeline(config)