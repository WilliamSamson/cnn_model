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


class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, image_dir, char_to_index, img_transform=None):
        self.df = pd.read_csv(csv_file)  # CSV with columns: 'label', 'image_name'
        self.image_dir = image_dir       # Directory where images are stored (e.g., 'Plates')
        self.char_to_index = char_to_index
        # Default transform converts PIL image to tensor and normalizes pixel values
        self.img_transform = img_transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 128))  # Ensure consistent image size (H x W) if needed
        ])

    def __len__(self):
        return len(self.df)

    def encode_label(self, label):
        # Convert label to sequence of indices
        return [self.char_to_index[c] for c in label if c in self.char_to_index]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        image_name = row['image_name']
        image_path = os.path.join(self.image_dir, image_name)
        # Open image and convert to RGB
        image = Image.open(image_path).convert("RGB")
        if self.img_transform:
            image = self.img_transform(image)
        encoded_label = self.encode_label(label)
        return image, torch.tensor(encoded_label, dtype=torch.long)

# ------------------------------
# 2. Data Module for PyTorch Lightning
# ------------------------------
class LicensePlateDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, image_dir, char_to_index, batch_size=16, img_transform=None):
        super().__init__()
        self.csv_train = csv_train
        self.csv_val = csv_val
        self.image_dir = image_dir
        self.char_to_index = char_to_index
        self.batch_size = batch_size
        self.img_transform = img_transform

    def setup(self, stage=None):
        self.train_dataset = LicensePlateDataset(self.csv_train, self.image_dir, self.char_to_index, self.img_transform)
        self.val_dataset = LicensePlateDataset(self.csv_val, self.image_dir, self.char_to_index, self.img_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# ------------------------------
# 3. CRNN Model with CTC Loss
# ------------------------------
class CRNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Assuming input images are resized to 64x128, after 3 poolings (factor 8 reduction in height)
        # The output feature map height becomes 8. The width will depend on the input size.
        self.lstm = nn.LSTM(128 * 8, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)
        self.ctc_loss = nn.CTCLoss(blank=num_classes - 1, reduction='mean', zero_infinity=True)

    def forward(self, x):
        x = self.cnn(x)
        # Permute to (batch, width, channels*height) for LSTM
        x = x.permute(0, 3, 1, 2).flatten(2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Set constant input length (depends on CNN architecture, here assumed 32 time-steps)
        input_lengths = torch.full((images.size(0),), 32, dtype=torch.long)
        # Compute label lengths dynamically from the provided labels
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

        predictions = self(images)
        loss = self.ctc_loss(predictions.log_softmax(2), labels, input_lengths, label_lengths)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


# Define the allowed characters and mapping
char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_index = {char: i for i, char in enumerate(char_list)}
num_classes = len(char_list) + 1  # +1 for CTC blank token


# Load Model & Scaler
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "Training_Set"/ "plates.csv"
data_path2 = project_root / "Training_Set"/  "Plates"

if not data_path.exists():
    raise FileNotFoundError(f"File not found: {data_path}")

if not data_path2.exists():
    raise FileNotFoundError(f"File not found: {data_path2}")



# Paths to your CSV files and image directory
csv_train =data_path  # Update with your training CSV path
csv_val = data_path    # Update with your validation CSV path
image_dir = data_path2 # Directory where images (img_1.jpeg ... img_100.jpeg) are stored

# Instantiate Data Module with optional image transforms
data_module = LicensePlateDataModule(csv_train, csv_val, image_dir, char_to_index, batch_size=16)

# Create the CRNN model
model = CRNN(num_classes)

# Setup PyTorch Lightning Trainer and start training
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, data_module)
