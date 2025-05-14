import os
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ComicsDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        """
        Dataset per a caption_relevance amb totes les imatges disponibles per caption.
        """
        base_path = "/home/datasets/ComicsPAP/caption_relevance"
        parquet_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith(split)]
        self.dataset = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        if max_samples:
            self.dataset = self.dataset.iloc[:max_samples]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        caption = example["previous_panel_caption"]
        options = example["options"]            
        label = example["solution_index"]      

        return {
            "texts": caption,
            "images": options,
            "label": label
        }
def comics_collate_fn(batch):
    """
    Agrupa una batch de mostres (caption + label).
    Retorna llistes: texts, llistes d’imatges (una per exemple), labels.
    """
    texts = [item["texts"] for item in batch]
    image_sets = [item["images"] for item in batch]  
    labels = [item["label"] for item in batch]
    return {
        "texts": texts,
        "image_sets": image_sets,
        "labels": labels
    }

def make_comics_loader(split="train", batch_size=4, max_samples=None, shuffle=True):
    dataset = ComicsDataset(split=split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=2,
                        collate_fn=comics_collate_fn,
                        pin_memory=True)
    return loader
