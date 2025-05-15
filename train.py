import torch
import wandb
from tqdm.auto import tqdm
import torch.nn.functional as F
import os

def compute_accuracy(sim_scores, labels):
    preds = torch.argmax(sim_scores, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def train_batch(model, batch, optimizer, device="cuda"):
    texts = batch["texts"]
    image_sets = batch["image_sets"]
    labels = batch["labels"]
    # print("Tipus de texts:", type(texts))
    # print("Mostra de texts:", texts[:2])

    # Codificació conjunta via forward
    text_embeddings, _ = model(raw_text=texts)  # [B, D]

    _, image_embeddings = zip(*[
        model(image_paths=img_list) for img_list in image_sets
    ])
    image_embeddings = torch.stack(image_embeddings).to(device)  # [B, K, D]

    sim_scores = torch.stack([
        model.compute_similarity(text_embeddings[i].unsqueeze(0), image_embeddings[i])
        for i in range(text_embeddings.size(0))
    ])  # [B, K]
    sim_scores = sim_scores.squeeze(1)


    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    loss = F.cross_entropy(sim_scores, labels_tensor)
    acc = compute_accuracy(sim_scores, labels_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc, len(labels)

def validate(model, loader, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            texts = batch["texts"]
            image_sets = batch["image_sets"]
            labels = batch["labels"]

            text_embeddings, _ = model(raw_text=texts)  # [B, D]

            _, image_embeddings = zip(*[
                model(image_paths=img_list) for img_list in image_sets
            ])
            image_embeddings = torch.stack(image_embeddings).to(device)  # [B, K, D]

            sim_scores = []
            for i in range(text_embeddings.size(0)):
                similarity = model.compute_similarity(
                    text_embeddings[i].unsqueeze(0),
                    image_embeddings[i]
                )
                sim_scores.append(similarity.squeeze(0))

            sim_scores = torch.stack(sim_scores)  # [B, K]

            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            loss = F.cross_entropy(sim_scores, labels_tensor)
            acc = compute_accuracy(sim_scores, labels_tensor)

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
        print(f"Checkpoint carregat des de {path}, reprenent a partir de l'època {checkpoint['epoch']+1}")
        return checkpoint['epoch'] + 1
    else:
        print("No s'ha trobat cap checkpoint. Començant des de zero.")
        return 0

def train(model, train_loader, val_loader, optimizer, config, device="cuda"):
    wandb.watch(model, log="all", log_freq=10)
    example_ct = 0

    # Carrega el checkpoint si existeix
    start_epoch = load_checkpoint(config["checkpoint_path"], model, optimizer)
    
    for epoch in tqdm(range(start_epoch, config.epochs), desc="Entrenant"):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        print(f"\n[INFO] ➡️ Inici de l'època {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            print(f"[INFO] 🚀 Processant batch {batch_idx} de l'època {epoch}...")

            loss, acc, n = train_batch(model, batch, optimizer, device)
            example_ct += n
            total_loss += loss * n
            total_acc += acc * n
            total_samples += n

            print(f"[INFO] ✅ Batch {batch_idx} completat. Loss: {loss:.4f}, Acc: {acc:.4f}")

        print(f"[INFO] 📝 Fi de l'època {epoch}. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples

        val_loss, val_acc = validate(model, val_loader, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }, step=example_ct)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
