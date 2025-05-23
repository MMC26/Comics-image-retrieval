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
# Dataset pre-processat LAZY (chunk per chunk)
# -------------------------------
class ChunkedPreprocessedComicsDataset(Dataset):
    def __init__(self, split="train", base_path="/home/datasets/ComicsPAP/caption_relevance_preprocessed_v3"):
        self.base_path = base_path
        self.split = split

        files = os.listdir(base_path)

        def extract_chunk_number(filename):
            m = re.search(r"chunk(\d+)", filename)
            if m:
                return int(m.group(1))
            else:
                raise ValueError(f"Fitxer no té format correcte de chunk: {filename}")

        self.chunk_files = sorted(
            [f for f in files if f.startswith(f"{split}_X_chunk") and f.endswith(".pt")],
            key=extract_chunk_number
        )

        print(f"[INFO] S'han trobat {len(self.chunk_files)} chunks per al split '{split}':")
        for f in self.chunk_files:
            print(f"  - {f}")

        self.chunk_lengths = []
        self.cumulative_lengths = []
        total_len = 0

        print("[INFO] Calculant longituds dels chunks (sense carregar tot el contingut)...")
        for f in self.chunk_files:
            path = os.path.join(base_path, f)
            data = torch.load(path, weights_only=True)
            length = data["labels"].shape[0]
            self.chunk_lengths.append(length)
            total_len += length
            self.cumulative_lengths.append(total_len)
            print(f"  - Chunk {f}: {length} mostres")

        print(f"[INFO] Longitud total dataset: {total_len} mostres")

        self._current_chunk_index = None
        self._current_chunk_data = None

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.cumulative_lengths[-1]:
            raise IndexError(f"Índex {idx} fora dels límits del dataset (0-{self.cumulative_lengths[-1] - 1})")

        chunk_idx = 0
        while chunk_idx < len(self.cumulative_lengths) and idx >= self.cumulative_lengths[chunk_idx]:
            chunk_idx += 1

        if chunk_idx == len(self.cumulative_lengths):
            raise IndexError(f"Índex {idx} fora dels límits del dataset.")

        prev_cum_len = 0 if chunk_idx == 0 else self.cumulative_lengths[chunk_idx - 1]
        local_idx = idx - prev_cum_len

        if self._current_chunk_index != chunk_idx:
            chunk_path = os.path.join(self.base_path, self.chunk_files[chunk_idx])
            print(f"[INFO] Carregant chunk {self.chunk_files[chunk_idx]} (index {chunk_idx}) a memòria...")
            self._current_chunk_data = torch.load(chunk_path, weights_only=True)
            self._current_chunk_index = chunk_idx

        item = {
            "input_ids": self._current_chunk_data["input_ids"][local_idx],
            "attention_mask": self._current_chunk_data["attention_masks"][local_idx],
            "image_set": self._current_chunk_data["image_sets"][local_idx],
            "label": self._current_chunk_data["labels"][local_idx]
        }

        if "texts" in self._current_chunk_data:
            item["text"] = self._current_chunk_data["texts"][local_idx]

        return item


def preprocessed_collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_masks": torch.stack([item["attention_mask"] for item in batch]),
        "image_sets": torch.stack([item["image_set"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "texts": [item["text"] for item in batch]  # Afegim texts per lots NOU
    }


# -------------------------------
# Loader selector
# -------------------------------
def make_comics_loader(split="train", batch_size=4, max_samples=None, shuffle=True, num_workers=2, preprocessed=False):
    if preprocessed:
        dataset = ChunkedPreprocessedComicsDataset(split=split)
        if max_samples:
            # dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))
            max_samples = min(max_samples, len(dataset))
            dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))
        collate = preprocessed_collate_fn
    else:
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


