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

def preprocess_image(img, device="cuda"):
    """No utilitzar aquesta mai pq ja ho fa el models
        """
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


def compute_and_save_test_image_embeddings(model, test_dataset, device="cuda", save_path="test_image_embeddings.pt"):
    """
    Calcula i desa els embeddings de totes les imatges del dataset de test.

    Args:
        model: model entrenat.
        test_dataset: dataset de test (objecte amb __getitem__ i __len__).
        device: dispositiu.
        save_path: fitxer on guardar embeddings i imatges.
    """
    model.eval()
    all_embeddings = []
    all_image_paths = []  # O llista d'imatges PIL (o el que tinguis) per després mostrar

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            # Suposo que cada exemple és un dict amb 'image_set' (llista d'imatges) i potser 'text' si vols
            example = test_dataset[i]
            image_set = example["image_set"]  # llista d'imatges (PIL, paths o tensors)

            # Preprocessa totes les imatges
            image_tensors = [preprocess_image(img, device) for img in image_set]
            image_batch = torch.stack(image_tensors)  # [K, C, H, W]

            # Embeddings de les imatges
            _, image_embs = model(image_paths=image_batch)
            all_embeddings.append(image_embs.cpu())
            all_image_paths.extend(image_set)  # Aquí guardem totes les imatges a nivell global

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N_imatges, D]

    # Guardem al disc
    torch.save({
        "embeddings": all_embeddings,
        "image_refs": all_image_paths
    }, save_path)

    print(f"[INFO] Guardats {all_embeddings.shape[0]} embeddings a {save_path}")


def load_test_image_embeddings(path="test_image_embeddings.pt", device="cuda"):
    data = torch.load(path, map_location=device)
    embeddings = data["embeddings"].to(device)
    image_refs = data["image_refs"]  # llista amb totes les imatges (paths o PIL)
    return embeddings, image_refs


def find_top_k_images_for_text(model, text, image_embeddings, image_refs, k=10, device="cuda"):
    model.eval()
    with torch.no_grad():
        text_emb, _ = model(raw_text=[text])
        text_emb = text_emb.to(device)  # [1, D]

        # Similitud text vs totes les imatges
        sim_scores = model.compute_similarity(text_emb, image_embeddings).squeeze(0)  # [N_imatges]

        topk_scores, topk_indices = torch.topk(sim_scores, k)

        topk_images = [image_refs[i] for i in topk_indices.cpu().tolist()]
        topk_scores = topk_scores.cpu().tolist()

        return topk_images, topk_scores


def display_images(images, scores=None, title_prefix=""):
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]

    for i, img in enumerate(images):
        if isinstance(img, str):
            pil_img = PILImage.open(img).convert("RGB")
        elif isinstance(img, PILImage.Image):
            pil_img = img
        elif isinstance(img, torch.Tensor):
            pil_img = PILImage.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        else:
            pil_img = PILImage.new("RGB", (100, 100))

        axs[i].imshow(pil_img)
        axs[i].axis("off")
        if scores is not None:
            axs[i].set_title(f"{title_prefix}Score: {scores[i]:.4f}", fontsize=9)

    plt.tight_layout()
    plt.show()


# Exemple d'ús al final del fitxer o en main:
if __name__ == "__main__":
    # Carregar model, dataset, etc.
    model = ...  # Instancia el teu model
    test_dataset = ...  # El teu dataset de test

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1. Calcular i guardar embeddings (només un cop)
    compute_and_save_test_image_embeddings(model, test_dataset, device=device, save_path="test_image_embeddings.pt")

    # 2. Carregar embeddings
    image_embeddings, image_refs = load_test_image_embeddings("test_image_embeddings.pt", device=device)

    # 3. Fer cerques
    query_text = "aquí el teu text de cerca"
    top_images, top_scores = find_top_k_images_for_text(model, query_text, image_embeddings, image_refs, k=10, device=device)

    # 4. Mostrar resultats
    display_images(top_images, top_scores, title_prefix="Similarity ")

def visualize_top10(model, test_loader, device="cuda"):
    """
    Mostra les 10 prediccions amb més confiança del conjunt de test.
    Usa el model i el test_loader ja carregats.
    """
    model.eval()
    scored_results = []

    with torch.no_grad():
        for batch in test_loader:
            for text, image_set in zip(batch["texts"], batch["image_sets"]):
                text_emb, _ = model(raw_text=[text])
                _, image_embs = model(image_paths=image_set)

                sim_scores = model.compute_similarity(text_emb, image_embs).squeeze(0)
                top_score, top_idx = torch.max(sim_scores, dim=0)

                scored_results.append({
                    "text": text,
                    "image": image_set[top_idx.item()],
                    "score": top_score.item()
                })

    top10 = sorted(scored_results, key=lambda x: x["score"], reverse=True)[:10]

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as PILImage

    fig, axs = plt.subplots(1, 10, figsize=(30, 5))
    for i, res in enumerate(top10):
        img = res["image"]
        img_np = np.array(img) / 255.0 if isinstance(img, PILImage.Image) else np.zeros((100, 100, 3))
        axs[i].imshow(img_np)
        axs[i].axis("off")
        axs[i].set_title(f"{res['score']:.2f}", fontsize=8)
        axs[i].text(0, -15, res["text"][:50] + ("..." if len(res["text"]) > 50 else ""), fontsize=6, wrap=True)

    plt.tight_layout()
    plt.show()
def denormalize_imgnet(tensor_img):
    """Reverteix la normalització ImageNet per visualització."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor_img * std + mean


def visualize_top10_live(model, test_loader, device="cuda", num_texts=5):
    """
    Per a cada exemple aleatori del test, mostra les 10 imatges més similars al text.
    Mostra les imatges amb el seu score, i imprimeix el text a terminal.
    """
    model.to(device)
    model.eval()

    all_images = []
    all_texts = []

    print("[INFO] Recollint totes les imatges i textos del test...")
    with torch.no_grad():
        for batch in test_loader:
            all_texts.extend(batch["texts"])
            for image_set in batch["image_sets"]:
                all_images.extend(image_set)

    print(f"[INFO] Total imatges: {len(all_images)} | Total textos: {len(all_texts)}")

    # Codificar imatges en batches
    print("[INFO] Codificant imatges en batches...")
    batch_size = 128
    image_emb_list = []
    for i in range(0, len(all_images), batch_size):
        batch_imgs = all_images[i:i+batch_size]
        _, batch_embs = model(image_paths=batch_imgs)
        image_emb_list.append(batch_embs.cpu())

    all_image_embs = torch.cat(image_emb_list, dim=0).to(device)

    # Seleccionar textos aleatoris
    random.shuffle(all_texts)
    selected_texts = all_texts[:num_texts]

    # Fer comparació i visualització
    with torch.no_grad():
        all_similarity_scores = []  # Per guardar totes les similituds
        for i, text in enumerate(selected_texts):
            # trimmed_text = " ".join(text.strip().split()[:100]) #he posat 100 pq es el que aga ara del text, CANVIAR SI CANVIEM EL MODEL
            # print(f"\n📝 TEXT {i+1} (100 paraules):\n{trimmed_text}")
            # text_emb, _ = model(raw_text=[text])
            if model.remove_first_sentence:
                cleaned_text = model.remove_first_sentence([text])[0]
                trimmed_text = " ".join(cleaned_text.strip().split()[:100])
            else:
                trimmed_text = " ".join(text.strip().split()[:100])

            print(f"\n📝 TEXT {i+1} (100 paraules que el model ha usat):\n{trimmed_text}")
            text_emb, _ = model(raw_text=[text])

            sim_scores = model.compute_similarity(text_emb, all_image_embs).squeeze(0)
            all_similarity_scores.extend(sim_scores.cpu().numpy().tolist())

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
        plt.figure(figsize=(8, 4))

        plt.hist(all_similarity_scores, bins=50, color='skyblue', edgecolor='black')
        plt.title("Distribució de similituds entre textos i imatges")
        plt.xlabel("Similitud (cosine score)")
        plt.ylabel("Freqüència")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def visualize_fixed_texts(model, image_loader, fixed_texts, device="cuda"):
    """
    Mostra les 10 imatges més similars a cada text de la llista `fixed_texts`.
    Mostra les imatges amb el seu score, imprimeix el text usat i genera un histograma de similituds.
    """
    model.to(device)
    model.eval()

    all_images = []

    print("[INFO] Recollint totes les imatges...")
    with torch.no_grad():
        for batch in image_loader:
            for image_set in batch["image_sets"]:
                all_images.extend(image_set)

    print(f"[INFO] Total imatges: {len(all_images)}")

    # Codificar imatges en batches
    print("[INFO] Codificant imatges en batches...")
    batch_size = 128
    image_emb_list = []
    for i in range(0, len(all_images), batch_size):
        batch_imgs = all_images[i:i+batch_size]
        _, batch_embs = model(image_paths=batch_imgs)
        image_emb_list.append(batch_embs.cpu())

    all_image_embs = torch.cat(image_emb_list, dim=0).to(device)

    # Comparació
    with torch.no_grad():
        all_similarity_scores = []

        for i, text in enumerate(fixed_texts):
            if model.remove_first_sentence:
                cleaned_text = model.remove_first_sentence([text])[0]
                trimmed_text = " ".join(cleaned_text.strip().split()[:100])
            else:
                trimmed_text = " ".join(text.strip().split()[:100])

            print(f"\n📝 TEXT {i+1} (100 paraules que el model ha usat):\n{trimmed_text}")
            text_emb, _ = model(raw_text=[text])

            sim_scores = model.compute_similarity(text_emb, all_image_embs).squeeze(0)
            all_similarity_scores.extend(sim_scores.cpu().numpy().tolist())

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

        plt.figure(figsize=(8, 4))

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

        print(f"\n📝 TEXT global (index {index}):\n{display_text}\n")

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
            print(f"⚠️ Exemple {i} fora del rang de validació.")
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

        print(f"🟩 Exemple {i}: Similitud correcta = {correct_score:.4f} — Posició = {correct_rank + 1}")
        if higher_scores:
            print(f"    Imatges més similars (incorrectes): {[round(s, 4) for s in higher_scores]}")
        else:
            print(f"    ✅ Cap imatge ha quedat per davant!")

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


def plot_global_recall_distribution(model, val_loader, device="cuda"):
    """
    Mostra la distribució de la posició en què apareix la imatge correcta
    per cada text dins del conjunt de validació global.
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

    # Codifica totes les imatges
    image_embeddings = []
    batch_size = 128
    for i in range(0, len(all_images), batch_size):
        batch_imgs = all_images[i:i+batch_size]
        _, embs = model(image_paths=batch_imgs)
        image_embeddings.append(embs.cpu())

    all_image_embs = torch.cat(image_embeddings, dim=0).to(device)

    # Calcular posicions
    position_counts = Counter()
    total = len(all_texts)

    for i, (text, correct_idx) in enumerate(zip(all_texts, correct_indices)):
        text_emb, _ = model(raw_text=[text])
        text_emb = text_emb.to(device)

        sim_scores = model.compute_similarity(text_emb, all_image_embs).squeeze(0).detach().cpu()
        sorted_indices = torch.argsort(sim_scores, descending=True)

        correct_rank = (sorted_indices == correct_idx).nonzero(as_tuple=True)[0].item()
        position_counts[correct_rank + 1] += 1  # +1 per tenir posicions 1-based

    # Crear llistes ordenades per visualització
    max_rank = max(position_counts.keys())
    positions = list(range(1, max_rank + 1))
    counts = [position_counts.get(pos, 0) for pos in positions]

    # Visualització
    plt.figure(figsize=(12, 5))
    plt.bar(positions, counts, color='skyblue')
    plt.title("Distribució de la posició de la imatge correcta")
    plt.xlabel("Posició en el rànquing (1 = top1)")
    plt.ylabel("Nombre de textos")
    plt.xticks(positions)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    print(f"[INFO] Processats {total} exemples.")
    for pos in range(1, min(11, max_rank + 1)):
        count = position_counts.get(pos, 0)
        print(f"🔹 Top {pos}: {count} textos ({count/total:.2%})")
