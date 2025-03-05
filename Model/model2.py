import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    # Keep labels as list for variable-length sequences (for CTCLoss)
    return images, list(labels)


class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, image_dir, char_to_index, img_transform=None):
        self.df = pd.read_csv(csv_file)  # Expects columns: 'label', 'image_name'
        self.image_dir = image_dir
        self.char_to_index = char_to_index
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def encode_label(self, label):
        # Encode each character into its corresponding index
        return [self.char_to_index[c] for c in label if c in self.char_to_index]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        image_name = row['image_name']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.img_transform:
            image = self.img_transform(image)
        encoded_label = self.encode_label(label)
        return image, torch.tensor(encoded_label, dtype=torch.long)


class LicensePlateDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, image_dir, char_to_index, batch_size=16, train_transform=None,
                 val_transform=None):
        super().__init__()
        self.csv_train = csv_train
        self.csv_val = csv_val
        self.image_dir = image_dir
        self.char_to_index = char_to_index
        self.batch_size = batch_size
        self.train_transform = train_transform or transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
        self.val_transform = val_transform or transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_dataset = LicensePlateDataset(self.csv_train, self.image_dir, self.char_to_index,
                                                 self.train_transform)
        self.val_dataset = LicensePlateDataset(self.csv_val, self.image_dir, self.char_to_index, self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=custom_collate, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=custom_collate, num_workers=4)


class CRNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # A deeper CNN architecture for richer feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # With input images resized to 64x128, three poolings yield a feature map of size 8x16.
        # Feature dimension per time step becomes 256 * 8 = 2048.
        self.lstm = nn.LSTM(256 * 8, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_classes)
        self.ctc_loss = nn.CTCLoss(blank=num_classes - 1, reduction='mean', zero_infinity=True)

    def forward(self, x):
        x = self.cnn(x)
        # x shape: (batch, channels, height, width) => (batch, 256, 8, 16)
        # Permute and flatten to (batch, width, channels*height)
        x = x.permute(0, 3, 1, 2).flatten(2)  # (batch, 16, 256*8)
        x, _ = self.lstm(x)  # (batch, 16, 512)
        x = self.fc(x)  # (batch, 16, num_classes)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        T = predictions.shape[1]
        input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        targets = torch.cat(labels)
        log_probs = predictions.log_softmax(2).transpose(0, 1)
        loss = self.ctc_loss(log_probs, targets, input_lengths, label_lengths)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        T = predictions.shape[1]
        input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        targets = torch.cat(labels)
        log_probs = predictions.log_softmax(2).transpose(0, 1)
        loss = self.ctc_loss(log_probs, targets, input_lengths, label_lengths)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]


def decode_prediction(pred, index_to_char, blank_idx):
    pred_texts = []
    for p in pred:
        p_idx = p.argmax(dim=1)  # (T,)
        prev = None
        decoded = []
        for idx in p_idx:
            idx = idx.item()
            if idx != blank_idx and idx != prev:
                decoded.append(index_to_char.get(idx, ''))
            prev = idx
        pred_texts.append("".join(decoded))
    return pred_texts


# Allowed characters and mappings
char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_index = {char: i for i, char in enumerate(char_list)}
num_classes = len(char_list) + 1  # +1 for the CTC blank token

project_root = Path(__file__).resolve().parents[1]
csv_file = project_root / "Training_Set" / "plates.csv"
image_dir = project_root / "Training_Set" / "Plates"

if not csv_file.exists():
    raise FileNotFoundError(f"File not found: {csv_file}")
if not image_dir.exists():
    raise FileNotFoundError(f"Directory not found: {image_dir}")

# Using the same CSV for training and validation for demonstration; consider proper splitting.
csv_train = str(csv_file)
csv_val = str(csv_file)

data_module = LicensePlateDataModule(csv_train, csv_val, str(image_dir), char_to_index, batch_size=16)
model = CRNN(num_classes)

# Extended training to 100 epochs to give the model more time to learn
trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1)
trainer.fit(model, data_module)

# Inference demo
index_to_char = {i: char for char, i in char_to_index.items()}
blank_idx = num_classes - 1

model.eval()
val_loader = data_module.val_dataloader()
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        predictions = model(images)  # (batch, T, num_classes)
        pred_texts = decode_prediction(predictions, index_to_char, blank_idx)
        true_texts = ["".join(index_to_char.get(i.item(), '') for i in label) for label in labels]
        for true, pred in zip(true_texts, pred_texts):
            print(f"True: {true} | Predicted: {pred}")
        break  # Process one batch for demonstration
