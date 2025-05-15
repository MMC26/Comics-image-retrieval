import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage


def preprocess_image(img, device="cuda"):
    if isinstance(img, str):
        img = PILImage.open(img).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)
    elif isinstance(img, PILImage.Image):
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)
    elif isinstance(img, torch.Tensor):
        if img.ndim == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        if img.max() > 1:
            img = img / 255.0
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    return img.to(device)


def test_demo(model, checkpoint_path, texts, image_sets, device="cuda", max_examples=5):
    """
    Predicts the best matching image for each text prompt using a trained model.

    Args:
        model: Your model class (instantiated but not yet loaded).
        checkpoint_path: Path to the trained model checkpoint (.pth).
        texts (list): List of strings (prompts).
        image_sets (list): List of image lists (per prompt).
        device (str): "cuda" or "cpu".
        max_examples (int): Max number of examples to display.
    """

    assert len(texts) == len(image_sets), "texts and image_sets must be the same length"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for idx, (text, image_set) in enumerate(zip(texts, image_sets)):
            print(f"\n--- Example {idx + 1} ---")
            print(f"TEXT: {text}")

            text_emb, _ = model(raw_text=[text])
            text_emb = text_emb.to(device)  # [1, D]

            # Preprocess and embed images
            image_tensors = [preprocess_image(img, device) for img in image_set]
            image_batch = torch.stack(image_tensors)  # [K, C, H, W]
            _, image_embs = model(image_paths=image_batch)

            # Compute similarities
            sim_scores = model.compute_similarity(text_emb, image_embs).squeeze(0)  # [K]
            predicted_idx = torch.argmax(sim_scores).item()

            print(f"Selected image: {predicted_idx + 1} / {len(image_set)}")

            # Plot images
            fig, axs = plt.subplots(1, len(image_set), figsize=(4 * len(image_set), 4))
            if len(image_set) == 1:
                axs = [axs]  # Handle single image case

            for j, img in enumerate(image_set):
                if isinstance(img, str):
                    img_np = np.array(PILImage.open(img).convert("RGB")) / 255.0
                elif isinstance(img, PILImage.Image):
                    img_np = np.array(img) / 255.0
                elif isinstance(img, torch.Tensor):
                    img_np = img.detach().cpu()
                    if img_np.shape[0] == 3:
                        img_np = img_np.permute(1, 2, 0)
                    img_np = img_np.numpy()
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                else:
                    img_np = np.zeros((100, 100, 3))  # fallback

                axs[j].imshow(img_np)
                axs[j].axis("off")
                if j == predicted_idx:
                    axs[j].set_title("SELECTED", color='green')

            plt.tight_layout()
            plt.show()

            results.append({
                "text": text,
                "predicted_index": predicted_idx,
                "similarity_scores": sim_scores.cpu().tolist()
            })

            if idx + 1 >= max_examples:
                break

    return results


def qualitative_test(model, test_loader, device="cuda", num_examples=5):
    model.eval()
    shown = 0

    with torch.no_grad():
        for batch in test_loader:
            texts = batch["texts"]
            image_sets = batch["image_sets"]  # Llista de llistes de paths

            text_embeddings, _ = model(raw_text=texts)
            _, image_embeddings = zip(*[model(image_paths=img_list) for img_list in image_sets])
            image_embeddings = torch.stack(image_embeddings).to(device)

            sim_scores = []
            for i in range(text_embeddings.size(0)):
                similarity = model.compute_similarity(
                    text_embeddings[i].unsqueeze(0),
                    image_embeddings[i]
                )
                sim_scores.append(similarity.squeeze(0))

            sim_scores = torch.stack(sim_scores)
            _, predicted = torch.max(sim_scores, dim=1)

            # Mostrem exemples
            for i in range(len(texts)):
                print(f"\nTEXT {i+1}: {texts[i]}")
                print(f"Imatge seleccionada: {predicted[i].item() + 1} / {len(image_sets[i])} imatges\n")

                # Dibuixem les imatges
                fig, axs = plt.subplots(1, len(image_sets[i]), figsize=(15, 3))
                for j, img_path in enumerate(image_sets[i]):
                    if isinstance(img_path, str):
                        # El cas on tens una ruta a una imatge
                        image = PILImage.open(img_path).convert("RGB")
                        image = np.array(image) / 255.0  # Normalitza a [0, 1]
                    elif isinstance(img_path, PILImage.Image):
                        # Ja és una imatge PIL
                        image = np.array(img_path) / 255.0
                    elif isinstance(img_path, torch.Tensor):
                        # És un tensor PyTorch, format [C,H,W], el passem a [H,W,C]
                        image = img_path.permute(1, 2, 0).cpu().numpy()
                    else:
                        print(f"Tipus no reconegut: {type(img_path)}")
                        continue

                    axs[j].imshow(image)
                    axs[j].axis('off')
                    if j == predicted[i]:
                        axs[j].set_title("SELECCIONADA", color='green')
                plt.tight_layout()
                plt.show()

                shown += 1
                if shown >= num_examples:
                    return
