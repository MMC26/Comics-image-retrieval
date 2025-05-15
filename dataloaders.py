import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import random
class ComicsDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        base_path = "/home/datasets/ComicsPAP/caption_relevance"
        parquet_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith(split)]
        self.dataset = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        self.dataset = self.dataset[self.dataset["previous_panel_caption"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        if max_samples:
            self.dataset = self.dataset.iloc[:max_samples]

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



def make_comics_loader(split="train", batch_size=4, max_samples=None, shuffle=True, num_workers=2):
    dataset = ComicsDataset(split=split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers,
                        collate_fn=comics_collate_fn,
                        pin_memory=True)
    return loader

