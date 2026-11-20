import time, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from dataset import find_dataset_path, load_splits, XRayDataset
from models import ClassicCNN, ModernResNet, AdvancedAttentionNet

warnings.filterwarnings("ignore")

DATA_DIR = find_dataset_path()
train_data, val_data, test_data = load_splits(DATA_DIR)
print(f"✅ Data loaded from: {DATA_DIR}")
print(f"📊 Splits: Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")

def evaluate_model(model_fn, name, seeds=[42, 100, 2026], lr=1e-3, augment=True, epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])
    eval_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    train_loader = DataLoader(XRayDataset(train_data, train_tf), batch_size=32, shuffle=True)
    test_loader = DataLoader(XRayDataset(test_data, eval_tf), batch_size=32)

    f1s, aucs, t_trains, t_infs, errors = [], [], [], [], []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = model_fn().to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        t0 = time.time()
        model.train()
        for _ in range(epochs):
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
        t_trains.append(time.time() - t0)

        model.eval()
        preds, probs, labels = [], [], []
        t_inf0 = time.time()
        with torch.no_grad():
            for x, y, fnames in test_loader:
                out = model(x.to(device))
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                pr = torch.argmax(out, dim=1).cpu().numpy()
                probs.extend(p); preds.extend(pr); labels.extend(y.numpy())
                if seed == seeds[0]:
                    for fn, yt, yp in zip(fnames, y.numpy(), pr):
                        if yt != yp and len(errors) < 3:
                            errors.append((fn, int(yt), int(yp)))
        t_infs.append((time.time() - t_inf0) / len(test_data) * 1000)
        f1s.append(f1_score(labels, preds, zero_division=0))
        aucs.append(roc_auc_score(labels, probs))

    params = sum(p.numel() for p in model.parameters())
    print(f"\n🔹 {name}")
    print(f"   F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"   Params: {params:,} | Size: {params*4/(1024**2):.2f} MB | Train: {np.mean(t_trains):.2f}s | Inf: {np.mean(t_infs):.2f} ms/img")
    return errors

def run_baseline():
    tf = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(), transforms.ToTensor()])
    X_tr, y_tr = zip(*[(x.view(-1).numpy(), y.item()) for x, y, _ in XRayDataset(train_data, tf)])
    X_te, y_te = zip(*[(x.view(-1).numpy(), y.item()) for x, y, _ in XRayDataset(test_data, tf)])
    t0 = time.time()
    clf = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
    t_tr = time.time() - t0
    t_inf0 = time.time()
    preds, probs = clf.predict(X_te), clf.predict_proba(X_te)[:, 1]
    t_inf = (time.time() - t_inf0) / len(y_te) * 1000
    print(f"\n🔹 Baseline (Logistic Regression)")
    print(f"   F1: {f1_score(y_te, preds):.4f} | AUC: {roc_auc_score(y_te, probs):.4f}")
    print(f"   Params: {clf.coef_.size+1:,} | Size: 0.01 MB | Train: {t_tr:.2f}s | Inf: {t_inf:.2f} ms/img")

if __name__ == "__main__":
    print(f"🖥️ Hardware: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    run_baseline()
    evaluate_model(ClassicCNN, "1. Classic CNN")
    evaluate_model(lambda: ModernResNet(False), "2a. Modern ResNet-18 (Scratch)")
    evaluate_model(lambda: ModernResNet(True), "2b. Modern ResNet-18 (Pretrained)")
    errs = evaluate_model(lambda: AdvancedAttentionNet(True), "3a. Advanced ResNet-18 + Custom Attention")
    evaluate_model(lambda: AdvancedAttentionNet(False), "3b. Ablation (No Attention)")
    evaluate_model(lambda: AdvancedAttentionNet(True), "4. Data Study (No Augmentation)", augment=False)
    print("\n❌ Error Examples:", errs)