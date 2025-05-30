from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import os
import io
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ComicsDataset(Dataset):
    def __init__(self, split="train", max_samples=None, transform=None):
        base_path = "/home/datasets/ComicsPAP/caption_relevance"
        parquet_files = sorted(
            [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith(split)]
        )

        data_chunks = []
        total = 0

        for f in parquet_files:
            df = pd.read_parquet(f)

            # Filter out rows with empty or non-string previous_panel_caption
            df = df[df["previous_panel_caption"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

            if max_samples is not None:
                needed = max_samples - total
                if needed <= 0:
                    break
                df = df.iloc[:needed]

            data_chunks.append(df)
            total += len(df)

            if max_samples is not None and total >= max_samples:
                break

        self.dataset = pd.concat(data_chunks, ignore_index=True)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset.iloc[idx]
        caption = example["previous_panel_caption"]

        options = example["options"]
        label = example["solution_index"]

        valid_options = []
        valid_indices = []
        for i, opt in enumerate(options):
            if isinstance(opt, dict) and "bytes" in opt and opt["bytes"]:
                image = Image.open(io.BytesIO(opt["bytes"]))
                if self.transform:
                    image = self.transform(image)
                valid_options.append(image)
                valid_indices.append(i)

        # Si és test (sense label), només tornem text i opcions
        if label == -1:
            return {
                "text": caption,
                "image_set": valid_options
            }

        # Per entrenament/validació normal
        if label not in valid_indices or len(valid_indices) < 2:
            return self.__getitem__((idx + 1) % len(self))

        pos_index = valid_indices.index(label)

        return {
            "text": caption,
            "image_set": valid_options,
            "label": pos_index
        }


def comics_collate_fn(batch):
    texts = [item["text"] for item in batch]
    image_sets = [item["image_set"] for item in batch]
    # Només agafa labels si existeixen
    if "label" in batch[0]:
        labels = [item["label"] for item in batch]
        return {
            "texts": texts,
            "image_sets": image_sets,
            "labels": labels
        }
    else:
        return {
            "texts": texts,
            "image_sets": image_sets
        }


# -------------------------------
# Loader selector
# -------------------------------
def make_comics_loader(split="train", batch_size=4, max_samples=None, shuffle=True, num_workers=2, preprocessed=False):
    
    transform = train_transform if split == "train" else val_transform
    dataset = ComicsDataset(split=split, max_samples=max_samples, transform=transform)
    collate = comics_collate_fn

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=collate,
                        pin_memory=True)
    return loader


