import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import torch

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
