import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DEBUG = True  # When True, only a small subset of the data is used


# --- UTILITY FUNCTIONS ---
def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(labels)


def visualize_samples(dataset, index_to_char, num_samples=5):
    # Visualize a few samples from the dataset to confirm data pipeline

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        image, label_tensor = dataset[i]
        label = "".join(index_to_char.get(c.item(), '') for c in label_tensor)
        axes[i].imshow(image.permute(1, 2, 0).numpy())
        axes[i].set_title(label)
        axes[i].axis("off")
    plt.show()


# --- DATASET & DATAMODULE ---
class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, image_dir, char_to_index, img_transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.char_to_index = char_to_index
        self.img_transform = img_transform or transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def encode_label(self, label):
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
        if DEBUG:
            # Use only a small subset to ensure overfitting capability
            small_idx = list(range(min(10, len(self.train_dataset))))
            self.train_dataset = Subset(self.train_dataset, small_idx)
            self.val_dataset = Subset(self.val_dataset, small_idx)
            print(f"[DEBUG] Overfitting mode: using {len(self.train_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=custom_collate, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=custom_collate, num_workers=2)


# --- MODEL DEFINITION ---
class CRNN(pl.LightningModule):
    def __init__(self, num_classes, use_simple_cnn=False):
        super().__init__()
        self.use_simple_cnn = use_simple_cnn
        if self.use_simple_cnn:
            # A simpler CNN to help overfit small dataset
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            cnn_output_dim = 32 * (64 // 2)  # height after pooling = 32; width will be handled as sequence length
        else:
            # A deeper CNN for feature extraction
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
            # For images resized to 64x128, three poolings give height=8.
            cnn_output_dim = 256 * 8

        self.lstm = nn.LSTM(cnn_output_dim, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_classes)
        self.ctc_loss = nn.CTCLoss(blank=num_classes - 1, reduction='mean', zero_infinity=True)

    def forward(self, x):
        x = self.cnn(x)
        # x: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2).flatten(2)  # (batch, width, channels*height)
        x, _ = self.lstm(x)
        x = self.fc(x)
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
        if batch_idx == 0:  # Log sample predictions
            self.log_predictions(predictions, labels)
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

    def log_predictions(self, predictions, labels):
        # Decode predictions to log a few sample outputs
        pred_texts = decode_prediction(predictions, self.trainer.datamodule.index_to_char, self.fc.out_features - 1)
        true_texts = ["".join(self.trainer.datamodule.index_to_char.get(i.item(), '') for i in label) for label in
                      labels]
        for t, p in zip(true_texts, pred_texts):
            self.logger.experiment.add_text("Prediction", f"True: {t} | Predicted: {p}", self.current_epoch)


def decode_prediction(pred, index_to_char, blank_idx):
    pred_texts = []
    for p in pred:
        p_idx = p.argmax(dim=1)
        prev = None
        decoded = []
        for idx in p_idx:
            idx = idx.item()
            if idx != blank_idx and idx != prev:
                decoded.append(index_to_char.get(idx, ''))
            prev = idx
        pred_texts.append("".join(decoded))
    return pred_texts


# --- MAIN EXECUTION ---
# Define allowed characters and mapping
char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_index = {char: i for i, char in enumerate(char_list)}
num_classes = len(char_list) + 1  # +1 for the CTC blank token
index_to_char = {i: char for char, i in char_to_index.items()}

# Set up paths using Pathlib
project_root = Path(__file__).resolve().parents[1]
csv_file = project_root / "Training_Set" / "plates.csv"
image_dir = project_root / "Training_Set" / "Plates"

if not csv_file.exists():
    raise FileNotFoundError(f"File not found: {csv_file}")
if not image_dir.exists():
    raise FileNotFoundError(f"Directory not found: {image_dir}")

csv_train = str(csv_file)
csv_val = str(csv_file)

data_module = LicensePlateDataModule(csv_train, csv_val, str(image_dir), char_to_index, batch_size=4)
# Expose index_to_char in datamodule for logging purposes
data_module.index_to_char = index_to_char

# Visualize a few samples to confirm data correctness
data_module.setup()
visualize_samples(data_module.train_dataset, index_to_char)

# You can toggle between a simpler or deeper CNN
model = CRNN(num_classes, use_simple_cnn=DEBUG)

trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1)
trainer.fit(model, data_module)

# Inference demo on debug subset
model.eval()
val_loader = data_module.val_dataloader()
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        predictions = model(images)
        pred_texts = decode_prediction(predictions, index_to_char, num_classes - 1)
        true_texts = ["".join(index_to_char.get(i.item(), '') for i in label) for label in labels]
        for true, pred in zip(true_texts, pred_texts):
            print(f"True: {true} | Predicted: {pred}")
        break
