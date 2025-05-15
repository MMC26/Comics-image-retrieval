"""
# Funció per calcular l'accuràcia
def compute_accuracy(sim_scores, labels):
    preds = torch.argmax(sim_scores, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

# Funció per realitzar el test
def test(model, test_loader, device="cuda", save=True, demo_examples=None):
    model.eval()  # Posem el model en mode d'inferència
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    # Realitzem inferències sense gradients (sense actualitzar pesos)
    with torch.no_grad():
        for batch in test_loader:
            texts = batch["texts"]
            image_sets = batch["image_sets"]
            labels = batch["labels"]

            # Càlcul de les embeddings per al text
            text_embeddings, _ = model(raw_text=texts)
            # Càlcul de les embeddings per a les imatges
            _, image_embeddings = zip(*[model(image_paths=img_list) for img_list in image_sets])
            image_embeddings = torch.stack(image_embeddings).to(device)  # [B, K, D]

            # Calcular la similaritat entre text i imatge
            sim_scores = torch.stack([
                model.compute_similarity(text_embeddings[i].unsqueeze(0), image_embeddings[i])
                for i in range(text_embeddings.size(0))
            ])  # [B, K]
            sim_scores = sim_scores.squeeze(1)  # Reduïm la dimensió

            # Convertim les etiquetes a tensor
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            # Comprovar si hi ha etiquetes amb valor -1
            if torch.any(labels_tensor == -1):
                print(f"Error! Etiquetes invàlides detectades: {labels_tensor[labels_tensor == -1]}")
                # Substituir el -1 per una etiqueta vàlida, per exemple, 0 (o una altra classe vàlida)
                labels_tensor[labels_tensor == -1] = 0  # Substitueix amb la classe que correspongui

            # Calculem la pèrdua i l'accuràcia
            loss = F.cross_entropy(sim_scores, labels_tensor)
            acc = compute_accuracy(sim_scores, labels_tensor)

            # Acumularem la pèrdua i l'accuràcia totals
            n = len(labels)
            total_loss += loss.item() * n
            total_acc += acc * n
            total_samples += n

    # Calculem la pèrdua i l'accuràcia mitjana
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    # Mostrem els resultats per pantalla
    print(f"\nTest Loss: {avg_loss:.4f} | Test Accuracy: {avg_acc:.4f}")
    # Loguem els resultats a wandb
    wandb.log({"test_loss": avg_loss, "test_accuracy": avg_acc})

    # Si volem exportar el model a ONNX
    if save:
        print("Exportant model a ONNX...")
        # Exemple de les dades per exportar
        dummy_text = ["Exemple de text de prova"]
        dummy_imgs = [["dummy/path/image1.png", "dummy/path/image2.png"]]  # assegura't que existeixen o mockeja
        text_input, _ = model(raw_text=dummy_text)
        _, image_input = model(image_paths=dummy_imgs[0])
        image_input = image_input.unsqueeze(0)  # [1, K, D]

        # Exportar el model a ONNX
        torch.onnx.export(model,
                          args=(dummy_text, dummy_imgs[0]),
                          f="model.onnx",
                          input_names=['text_input', 'image_input'],
                          output_names=['similarity_scores'],
                          dynamic_axes={'text_input': {0: 'batch_size'},
                                        'image_input': {0: 'batch_size'},
                                        'similarity_scores': {0: 'batch_size'}},
                          opset_version=10)
        wandb.save("model.onnx")

    # Exemple de prediccions amb textos i imatges de prova (demo)
    if demo_examples:
        print("\nExecutant exemples de demo:")
        for example in demo_examples:
            text = example["text"]
            imgs = example["images"]
            text_emb, _ = model(raw_text=[text])
            _, image_emb = model(image_paths=imgs)
            similarity = model.compute_similarity(text_emb, image_emb.unsqueeze(0))
            pred = torch.argmax(similarity, dim=1).item()
            print(f"Text: {text}")
            print(f"Predicció: Imatge index {pred} (de {len(imgs)})\n")

"""
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
