# train_sign_alphabet.py
# Treinamento de classificador de alfabeto manual (A–Z) usando ResNet18.
# Salva modelo e classes em artifacts/.

import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def main():
    # === CONFIGURAÇÕES ===
    DATA_DIR = "data/sign_es"   # pasta local: subpastas A/, B/, ..., Z/
    OUT_DIR = "artifacts"
    os.makedirs(OUT_DIR, exist_ok=True)

    IMG_SIZE   = 224
    BATCH_SIZE = 32
    EPOCHS     = 12
    LR         = 1e-3
    VAL_SPLIT  = 0.15
    SEED       = 42

    random.seed(SEED)
    torch.manual_seed(SEED)

    # === TRANSFORMAÇÕES ===
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # === DATASET / SPLIT ===
    base_ds = datasets.ImageFolder(DATA_DIR, transform=None)
    class_names = base_ds.classes
    print("Classes detectadas:", class_names)

    # salva classes para a inferência
    with open(Path(OUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    indices = list(range(len(base_ds)))
    labels  = [base_ds.samples[i][1] for i in indices]

    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=SEED, stratify=labels
    )

    # Dois datasets com transforms distintos
    train_full = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
    val_full   = datasets.ImageFolder(DATA_DIR, transform=val_tfms)

    # Subsets aplicando índices
    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)

    # === DATALOADERS ===
    # No Windows, use num_workers=0
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

    # === MODELO ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    # === LOSS / OPT ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    def avaliar(dl):
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return (loss_sum / total) if total else 0.0, (correct / total) if total else 0.0

    # === TREINAMENTO ===
    melhor_acc = 0.0
    melhor_modelo = Path(OUT_DIR, "best_resnet18.pt")

    for epoca in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = avaliar(val_dl)
        print(f"Época {epoca:02d}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > melhor_acc:
            melhor_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": class_names}, melhor_modelo)

    print("Treinamento finalizado. Melhor acurácia:", melhor_acc)
    print("Modelo salvo em:", melhor_modelo)

    # === AVALIAÇÃO FINAL DETALHADA (opcional) ===
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    if len(y_true) > 0:
        print("\n=== Relatório de Classificação (val) ===")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

        cm = confusion_matrix(y_true, y_pred)
        np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), cm)
        print("Matriz de confusão salva em artifacts/confusion_matrix.npy")


if __name__ == "__main__":
    # Necessário no Windows para multiprocessos
    import multiprocessing
    multiprocessing.freeze_support()
    main()
