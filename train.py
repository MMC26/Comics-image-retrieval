import random
import torch
import wandb
from tqdm.auto import tqdm
import torch.nn.functional as F
import os


def compute_accuracy(sim_scores, labels):
    preds = torch.argmax(sim_scores, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def train_batch(model, batch, optimizer, criterion, device="cuda"):
    texts = batch["texts"]
    image_sets = batch["image_sets"]
    labels = batch["labels"]

    text_embeddings, _ = model(raw_text=texts)

    image_embeddings = []
    for img_list in image_sets:
        _, img_emb = model(image_paths=img_list)
        image_embeddings.append(img_emb)
    image_embeddings = torch.stack(image_embeddings).to(device)  # [B, K, D]

    B, K, D = image_embeddings.shape
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    positives = image_embeddings[torch.arange(B), labels_tensor]  # [B, D]

    # Agafem només UN negatiu per cada exemple
    negative_indices = []
    for i in range(B):
        negs = [j for j in range(K) if j != labels[i]]
        neg_idx = random.choice(negs)
        negative_indices.append(neg_idx)
    negatives = image_embeddings[torch.arange(B), torch.tensor(negative_indices, device=device)]  # [B, D]

    loss = criterion(text_embeddings, positives, negatives)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Accuracy: compara quin és més proper
    sim_pos = F.cosine_similarity(text_embeddings, positives, dim=1)
    sim_neg = F.cosine_similarity(text_embeddings, negatives, dim=1)
    acc = (sim_pos > sim_neg).float().mean().item()

    return loss.item(), acc, len(labels)



def validate(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            texts = batch["texts"]
            image_sets = batch["image_sets"]
            labels = batch["labels"]

            text_embeddings, _ = model(raw_text=texts)

            image_embeddings = []
            for img_list in image_sets:
                _, img_emb = model(image_paths=img_list)
                image_embeddings.append(img_emb)
            image_embeddings = torch.stack(image_embeddings).to(device)

            B, K, D = image_embeddings.shape
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            positives = image_embeddings[torch.arange(B), labels_tensor]  # [B, D]

            # gafem només UN negatiu per cada exemple
            negative_indices = []
            for i in range(B):
                negs = [j for j in range(K) if j != labels[i]]
                neg_idx = random.choice(negs)
                negative_indices.append(neg_idx)
            negatives = image_embeddings[torch.arange(B), torch.tensor(negative_indices, device=device)]  # [B, D]

            loss = criterion(text_embeddings, positives, negatives)

            # Accuracy: compara quin és més proper
            sim_pos = F.cosine_similarity(text_embeddings, positives, dim=1)
            sim_neg = F.cosine_similarity(text_embeddings, negatives, dim=1)
            acc = (sim_pos > sim_neg).float().mean().item()

            n = len(labels)
            total_loss += loss.item() * n
            total_acc += acc * n
            total_samples += n

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc



def save_checkpoint(path, model, optimizer, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint carregat des de {path}, reprenent a partir de l'època {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1
    else:
        print("No s'ha trobat cap checkpoint. Començant des de zero.")
        return 0


def train(model, train_loader, val_loader, optimizer, criterion,config, device="cuda"):
    wandb.watch(model, log="all", log_freq=10)
    example_ct = 0

    # Carrega el checkpoint si existeix
    start_epoch = load_checkpoint(config["checkpoint_path"], model, optimizer)

    for epoch in tqdm(range(start_epoch, config["epochs"]), desc="Entrenant"):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        print(f"\n[INFO] ➡️ Inici de l'època {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            print(f"[INFO] 🚀 Processant batch {batch_idx} de l'època {epoch}...")

            loss, acc, n = train_batch(model, batch, optimizer,criterion, device)
            example_ct += n
            total_loss += loss * n
            total_acc += acc * n
            total_samples += n

            print(f"[INFO] ✅ Batch {batch_idx} completat. Loss: {loss:.4f}, Acc: {acc:.4f}")

        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples
        print(f"[INFO] 📝 Fi de l'època {epoch}. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc = validate(model, val_loader, criterion,device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }, step=example_ct)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        save_checkpoint(config["checkpoint_path"], model, optimizer, epoch)

 