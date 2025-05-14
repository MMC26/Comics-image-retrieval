import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io

class ComicsDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        base_path = "/home/datasets/ComicsPAP/caption_relevance"
        parquet_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith(split)]
        self.dataset = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        if max_samples:
            self.dataset = self.dataset.iloc[:max_samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        start_idx = idx
        while True:
            example = self.dataset.iloc[idx]
            caption = example["previous_panel_caption"]

            # Si no hi ha caption vàlid, passa al següent
            if not isinstance(caption, str) or not caption.strip():
                idx = (idx + 1) % len(self)
                if idx == start_idx:
                    raise ValueError("No hi ha cap exemple amb caption vàlid.")
                continue

            options = example["options"]
            label = example["solution_index"]

            valid_options = []
            valid_indices = []
            for i, opt in enumerate(options):
                if isinstance(opt, dict) and "bytes" in opt and opt["bytes"]:
                    # Carrega la imatge des de bytes
                    image = Image.open(io.BytesIO(opt["bytes"]))
                    valid_options.append(image)
                    valid_indices.append(i)

            # Si el label no apunta a una opció vàlida, passa al següent
            if label not in valid_indices:
                idx = (idx + 1) % len(self)
                if idx == start_idx:
                    raise ValueError("No hi ha cap exemple amb opcions vàlides.")
                continue

            new_label = valid_indices.index(label)
            return {
                "texts": caption,
                "images": valid_options,
                "label": new_label
            }
def comics_collate_fn(batch):
    texts = [item["texts"] for item in batch]
    image_sets = [item["images"] for item in batch]
    labels = [item["label"] for item in batch]
    return {"texts": texts, "image_sets": image_sets, "labels": labels}

def make_comics_loader(split="train", batch_size=4, max_samples=None, shuffle=True,num_workers=2):
    dataset = ComicsDataset(split=split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers,
                        collate_fn=comics_collate_fn,
                        pin_memory=True)
    return loader