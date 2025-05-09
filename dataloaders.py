from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ComicsDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        """
        Dataset per a caption_relevance amb totes les imatges disponibles per caption.
        """
        self.dataset = load_dataset("VLR-CVC/ComicsPAP", "caption_relevance", split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        caption = example["previous_panel_caption"]
        options = example["options"]            
        label = example["solution_index"]      

        return {
            "text": caption,
            "images": options,
            "label": label
        }
def comics_collate_fn(batch):
    """
    Agrupa una batch de mostres (caption + label).
    Retorna llistes: texts, llistes d’imatges (una per exemple), labels.
    """
    texts = [item["text"] for item in batch]
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
