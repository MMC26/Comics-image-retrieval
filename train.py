import random
import torch
import wandb
from tqdm.auto import tqdm
import torch.nn.functional as F
import os
from torchvision import transforms

def compute_recall_at_k(similarity_scores, labels, k=1):
    max_k = min(k, similarity_scores.shape[1])  # Protegeix contra errors si K < k
    topk = torch.topk(similarity_scores, k=max_k, dim=1).indices  # [B, k]
    labels = labels.unsqueeze(1)  # [B, 1]
    correct = (topk == labels).any(dim=1).float()
    return correct.mean().item()

def train_batch(model, batch, optimizer, criterion, device="cuda"):
    texts = batch["texts"]
    image_sets = batch["image_sets"]
    labels = batch["labels"]

    text_embeddings, _ = model(raw_text=texts)

    flat_image_paths = [path for img_list in image_sets for path in img_list]
    _, flat_img_embs = model(image_paths=flat_image_paths)

    B = len(image_sets)
    K = len(image_sets[0])
    image_embeddings = flat_img_embs.view(B, K, -1).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    positives = image_embeddings[torch.arange(B), labels_tensor]
    similarities = torch.bmm(image_embeddings, text_embeddings.unsqueeze(2)).squeeze(2)

    mask = torch.ones_like(similarities, dtype=torch.bool)
    mask[torch.arange(B), labels_tensor] = False
    masked_similarities = similarities.masked_fill(~mask, float('-inf'))
    hardest_neg_indices = masked_similarities.argmax(dim=1)
    negatives = image_embeddings[torch.arange(B), hardest_neg_indices]

    loss = criterion(text_embeddings, positives, negatives)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    recall1 = compute_recall_at_k(similarities, labels_tensor, k=1)
    return loss.item(), recall1, len(labels)

def validate(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_recall = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="🔍 Validating", leave=False)):
            texts = batch["texts"]
            image_sets = batch["image_sets"]
            labels = batch["labels"]

            text_embeddings, _ = model(raw_text=texts)
            flat_image_paths = [path for img_list in image_sets for path in img_list]
            _, flat_img_embs = model(image_paths=flat_image_paths)

            B = len(image_sets)
            K = len(image_sets[0])
            image_embeddings = flat_img_embs.view(B, K, -1).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            positives = image_embeddings[torch.arange(B), labels_tensor]
            sim_matrix = torch.bmm(image_embeddings, text_embeddings.unsqueeze(2)).squeeze(2)

            recall1 = compute_recall_at_k(sim_matrix, labels_tensor, k=1)

            sim_matrix.scatter_(1, labels_tensor.unsqueeze(1), float('-inf'))
            hardest_neg_idx = sim_matrix.argmax(dim=1)
            negatives = image_embeddings[torch.arange(B), hardest_neg_idx]

            loss = criterion(text_embeddings, positives, negatives)

            n = len(labels)
            total_loss += loss.item() * n
            total_recall += recall1 * n
            total_samples += n

    avg_loss = total_loss / total_samples
    avg_recall = total_recall / total_samples
    return avg_loss, avg_recall

def validate_global_retrieval(model, val_loader, criterion, device="cuda"):
    model.eval()
    model.to(device)

    all_images = []
    all_texts = []
    correct_image_indices = []

    print("[INFO] Recol·lectant totes les imatges i textos del val_loader...")

    with torch.no_grad():
        idx_counter = 0
        for batch in tqdm(val_loader, desc="🔄 Recol·lectant val"):
            for text, img_set, label in zip(batch["texts"], batch["image_sets"], batch["labels"]):
                all_texts.append(text)
                all_images.extend(img_set)
                correct_image_indices.append(idx_counter + label)
                idx_counter += len(img_set)

    print(f"[INFO] Total imatges: {len(all_images)} | Total textos: {len(all_texts)}")

    all_image_embs = []
    for i in range(0, len(all_images), 64):
        batch_imgs = all_images[i:i+64]
        _, img_embs = model(image_paths=batch_imgs)
        all_image_embs.append(img_embs)
    all_image_embs = torch.cat(all_image_embs, dim=0)

    text_embs, _ = model(raw_text=all_texts)

    similarity = torch.matmul(text_embs, all_image_embs.T)
    labels_tensor = torch.tensor(correct_image_indices, dtype=torch.long, device=device)

    recall1 = compute_recall_at_k(similarity, labels_tensor, k=1)
    recall10= compute_recall_at_k(similarity, labels_tensor, k=10)

    # Calcular la pèrdua mitjana amb cosine triplet loss
    positives = all_image_embs[labels_tensor]
    sim_matrix = torch.matmul(text_embs, all_image_embs.T)
    sim_matrix.scatter_(1, labels_tensor.unsqueeze(1), float('-inf'))
    hardest_neg_idx = sim_matrix.argmax(dim=1)
    negatives = all_image_embs[hardest_neg_idx]

    loss = criterion(text_embs, positives, negatives).item()

    print(f"[GLOBAL VAL] Recall@1: {recall1:.4f} | Loss: {loss:.4f}")
    print(f"[GLOBAL VAL] Recall@10: {recall10:.4f} | Loss: {loss:.4f}")
    return loss, recall1, recall10

def save_checkpoint(path, model, optimizer, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"[INFO] Checkpoint guardat a {path}")

def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[INFO] Checkpoint carregat des de {path}, reprenent a partir de l'època {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1
    else:
        print("[INFO] No s'ha trobat cap checkpoint. Començant des de zero.")
        return 0

def train(model, train_loader, val_loader, optimizer, criterion, config, device="cuda"):
    wandb.watch(model, log="all", log_freq=10)
    example_ct = 0

    #start_epoch = load_checkpoint(config["checkpoint_path"], model, optimizer)

    #for epoch in tqdm(range(start_epoch, config["epochs"]), desc="Entrenant"):
    for epoch in tqdm(range( config["epochs"]), desc="Entrenant"):
        model.train()
        total_loss = 0.0
        total_recall = 0.0
        total_samples = 0

        print(f"\n[INFO] ➔ Inici de l'època {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            print(f"[INFO] Processant batch {batch_idx} de l'època {epoch}...")

            loss, recall1, n = train_batch(model, batch, optimizer, criterion, device)
            example_ct += n
            total_loss += loss * n
            total_recall += recall1 * n
            total_samples += n

            print(f"[INFO] Batch {batch_idx} completat. Loss: {loss:.4f}, Recall@1: {recall1:.4f}")

        train_loss = total_loss / total_samples
        train_recall = total_recall / total_samples
        print(f"[INFO] Fi de l'època {epoch}. Train Loss: {train_loss:.4f}, Train Recall@1: {train_recall:.4f}")

        val_loss, val_recall = validate(model, val_loader, criterion, device)
        val_global_loss, val_global_recall1, val_global_recall10 = validate_global_retrieval(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_recall@1": train_recall,
            "val_loss": val_loss,
            "val_recall@1": val_recall,
            "val_global_loss": val_global_loss,
            "val_global_recall@1": val_global_recall1,
            "val_global_recall@10": val_global_recall10

        }, step=example_ct)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Recall@1: {train_recall:.8f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Recall@1: {val_recall:.8f}")
        print(f"[Epoch {epoch}] Val Global Loss: {val_global_loss:.4f} | Recall@1: {val_global_recall1:.8f}")
        print(f"[Epoch {epoch}] Val Global Loss: {val_global_loss:.4f} | Recall@10: {val_global_recall10:.8f}")


        #save_checkpoint(config["checkpoint_path"], model, optimizer, epoch) 