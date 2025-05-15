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
        example = self.dataset.iloc[idx]
        caption = example["previous_panel_caption"]

        if not isinstance(caption, str) or not caption.strip():
            return self.__getitem__((idx + 1) % len(self))  # recursivament salta

        options = example["options"]
        label = example["solution_index"]

        valid_options = []
        valid_indices = []
        for i, opt in enumerate(options):
            if isinstance(opt, dict) and "bytes" in opt and opt["bytes"]:
                image = Image.open(io.BytesIO(opt["bytes"]))
                valid_options.append(image)
                valid_indices.append(i)

        if label not in valid_indices or len(valid_indices) < 2:
            return self.__getitem__((idx + 1) % len(self))

        # Anchor = caption, Positive = correct image, Negative = any incorrect image
        pos_img = valid_options[valid_indices.index(label)]

        # Escull negatiu diferent del positiu
        neg_candidates = [img for i, img in enumerate(valid_options) if valid_indices[i] != label]
        neg_img = random.choice(neg_candidates)

        return {
            "text": caption,
            "positive": pos_img,
            "negative": neg_img,
        }

def comics_triplet_collate_fn(batch):
    texts = [item["text"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]
    return {"texts": texts, "positives": positives, "negatives": negatives}


def make_comics_loader(split="train", batch_size=4, max_samples=None, shuffle=True, num_workers=2):
    dataset = ComicsDataset(split=split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers,
                        collate_fn=comics_triplet_collate_fn,
                        pin_memory=True)
    return loader
