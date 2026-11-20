import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

def find_dataset_path():
    search_roots = [os.path.expanduser("~\\Downloads"), os.path.expanduser("~\\Desktop"), "C:\\", "D:\\"]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if "train" in dirnames and "test" in dirnames:
                if "coronahack" in dirpath.lower() or "xray" in dirpath.lower():
                    return dirpath
            if dirpath.count(os.sep) - root.count(os.sep) > 3:
                del dirnames[:]
    return None

def load_splits(data_dir):
    def get_files(sub):
        samples = []
        for root, _, files in os.walk(os.path.join(data_dir, sub)):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = 0 if "normal" in f.lower() or "normal" in root.lower() else 1
                    samples.append((os.path.join(root, f), label, f))
        return samples

    train_val = get_files("train")
    test_data = get_files("test")
    random.seed(42)
    random.shuffle(train_val)
    random.shuffle(test_data)
    return train_val[:240], train_val[240:320], test_data[:120]

class XRayDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, fname = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), fname