import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch
from collections import Counter


def denormalize_imgnet(tensor_img):
    """Reverteix la normalització ImageNet per visualització."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor_img * std + mean


def visualize_fixed_texts_from_loader(model, test_loader, device="cuda", indices=[5, 12, 20]):
    """
    Mostra les 10 imatges més similars per a textos en les posicions globals donades dins el test_loader.
    Si use_sentence_bert=True, mostra només les 100 primeres paraules
    i, si remove_first_sentence=True, elimina també la primera frase.
    """
    model.to(device)
    model.eval()

    # 1. Recollim els textos seleccionats pels índexs globals
    selected_texts = {}
    global_idx = 0
    for batch in test_loader:
        for text in batch["texts"]:
            if global_idx in indices:
                selected_texts[global_idx] = text
                if len(selected_texts) == len(indices):
                    break
            global_idx += 1
        if len(selected_texts) == len(indices):
            break

    if len(selected_texts) < len(indices):
        print(f"[WARN] No s'han trobat tots els índexs. Només s'han trobat: {list(selected_texts.keys())}")

    # 2. Recollim totes les imatges del test
    all_images = []
    for b in test_loader:
        for image_set in b["image_sets"]:
            all_images.extend(image_set)

    print(f"[INFO] Total imatges recollides: {len(all_images)}")

    # 3. Codifiquem totes les imatges
    batch_size = 128
    image_emb_list = []
    for i in range(0, len(all_images), batch_size):
        batch_imgs = all_images[i:i+batch_size]
        _, batch_embs = model(image_paths=batch_imgs)
        image_emb_list.append(batch_embs.cpu())

    all_image_embs = torch.cat(image_emb_list, dim=0).to(device)


    for index in indices:
        raw_text = selected_texts.get(index, None)
        if raw_text is None:
            print(f"[SKIP] No s'ha trobat text per l'índex {index}")
            continue

        display_text = raw_text

        if getattr(model, "use_sentence_bert", False) or getattr(model, "use_gtr", False):
            if getattr(model, "remove_first_sentence_flag", False):
                sentences = raw_text.strip().split(".")
                rest = ".".join(sentences[1:]).strip()
                display_text = rest if rest else sentences[0]

            display_text = " ".join(display_text.strip().split()[:100])

        print(f"\n TEXT global (index {index}):\n{display_text}\n")

        text_emb, _ = model(raw_text=[raw_text])
        sim_scores = model.compute_similarity(text_emb, all_image_embs).squeeze(0)
        top_scores, top_indices = torch.topk(sim_scores, k=10)

        fig, axs = plt.subplots(1, 10, figsize=(30, 4))
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]

        for j, idx in enumerate(top_indices):
            img = all_images[idx.item()]
            if isinstance(img, torch.Tensor):
                img = denormalize_imgnet(img.cpu())
                img_np = img.permute(1, 2, 0).numpy().clip(0, 1)
            elif isinstance(img, PILImage.Image):
                img_np = np.array(img) / 255.0
            else:
                img_np = np.zeros((224, 224, 3))

            axs[j].imshow(img_np)
            axs[j].axis("off")
            axs[j].set_title(f"{top_scores[j]:.2f}", fontsize=9)

        plt.tight_layout()
        plt.show()



def analyze_global_validation_similarities(model, val_loader, device="cuda", fixed_indices=[2, 20, 200]):
    """
    Analitza sempre les mateixes mostres de validació (per defecte: 0, 1, 2).
    Per cada text, calcula la similitud amb totes les imatges del conjunt de validació i mostra
    la posició de la imatge correcta dins del rànquing. També imprimeix les imatges incorrectes
    amb puntuació superior.
    """

    model.eval()
    model.to(device)

    all_texts = []
    all_images = []
    correct_indices = []

    idx_counter = 0
    for batch in val_loader:
        for text, image_set, label in zip(batch["texts"], batch["image_sets"], batch["labels"]):
            all_texts.append(text)
            all_images.extend(image_set)
            correct_indices.append(idx_counter + label)
            idx_counter += len(image_set)

    image_embeddings = []
    batch_size = 128
    for i in range(0, len(all_images), batch_size):
        batch_imgs = all_images[i:i+batch_size]
        _, embs = model(image_paths=batch_imgs)
        image_embeddings.append(embs.cpu())

    all_image_embs = torch.cat(image_embeddings, dim=0).to(device)

    for i in fixed_indices:
        if i >= len(all_texts):
            print(f" Exemple {i} fora del rang de validació.")
            continue

        text = all_texts[i]
        correct_idx = correct_indices[i]

        text_emb, _ = model(raw_text=[text])
        text_emb = text_emb.to(device)

        sim_scores = model.compute_similarity(text_emb, all_image_embs).squeeze(0).detach().cpu()
        sorted_scores, sorted_indices = torch.sort(sim_scores, descending=True)

        correct_rank = (sorted_indices == correct_idx).nonzero(as_tuple=True)[0].item()
        correct_score = sim_scores[correct_idx].item()

        higher_scores = sorted_scores[:correct_rank].numpy().tolist()

        print(f" Exemple {i}: Similitud correcta = {correct_score:.4f} — Posició = {correct_rank + 1}")
        if higher_scores:
            print(f"    Imatges més similars (incorrectes): {[round(s, 4) for s in higher_scores]}")
        else:
            print(f"     Cap imatge ha quedat per davant!")

        # Gràfic
        plt.figure(figsize=(10, 4))
        bars = plt.bar(range(len(sim_scores)), sorted_scores.numpy(), color="gray")
        bars[correct_rank].set_color("green")
        plt.axhline(correct_score, linestyle="--", color="green", alpha=0.6)

        plt.title(f"[GLOBAL] Exemple {i} - Posició correcta: {correct_rank + 1}/{len(sim_scores)}")
        plt.xlabel("Índex ordenat per similitud")
        plt.ylabel("Similitud (model.compute_similarity)")
        plt.tight_layout()
        plt.show()


