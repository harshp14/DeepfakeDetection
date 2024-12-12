import os
import torch
import cv2
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Preprocessing function
def preprocess_videos(video_paths, labels, sequence_length, transform, save_path):
    """
    Preprocess videos, convert to tensors, and save to disk.
    Args:
        video_paths (list): List of video file paths.
        labels (list): Corresponding labels for each video.
        sequence_length (int): Number of frames to sample per video.
        transform (callable): Transform to apply to each frame.
        save_path (str): Path to save preprocessed tensors.
    """
    data = []
    for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths), desc="Processing Videos"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if transform:
                frame = transform(frame)
            frames.append(frame)
        cap.release()

        # Ensure we have at least sequence_length frames
        if len(frames) >= sequence_length:
            frames = frames[:sequence_length]
        else:
            # Pad with zero frames if fewer frames
            padding = [torch.zeros_like(frames[0]) for _ in range(sequence_length - len(frames))]
            frames.extend(padding)

        frames_tensor = torch.stack(frames)  # Shape: (sequence_length, C, H, W)
        data.append((frames_tensor, label))

    # Save preprocessed data
    torch.save(data, save_path)
    print(f"Preprocessed data saved to {save_path}")

# Transform for ResNet preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load video paths and labels from text files
def load_video_data(source_folder):
    real_videos = []
    fake_videos = []

    with open(os.path.join(source_folder, "realVideos.txt"), "r") as f:
        real_videos = [os.path.join(source_folder, line.strip()) for line in f.readlines()]

    with open(os.path.join(source_folder, "fakeVideos.txt"), "r") as f:
        fake_videos = [os.path.join(source_folder, line.strip()) for line in f.readlines()]

    video_paths = real_videos + fake_videos
    labels = [1] * len(real_videos) + [0] * len(fake_videos)
    return video_paths, labels

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class PreprocessedVideoDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            featutes: 
            labels : 
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frames_tensor, label = self.features[idx], self.labels[idx]
        return frames_tensor, label

import torch.nn as nn
from torchvision.models import resnet18

class CNN_LSTM(nn.Module):
    def __init__(self, cnn_output_size=512, lstm_hidden_size=2048, lstm_num_layers=3, output_size=2):
        super(CNN_LSTM, self).__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove classification head

        self.lstm = nn.LSTM(cnn_output_size, lstm_hidden_size, lstm_num_layers, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        cnn_features = []

        for t in range(seq_len):
            cnn_output = self.cnn(x[:, t, :, :, :])  # Process each frame
            cnn_features.append(cnn_output)

        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, cnn_output_size)
        lstm_out, _ = self.lstm(cnn_features)  # Pass through LSTM
        lstm_out = self.dropout(lstm_out[:, -1, :])
        # output = self.fc(lstm_out[:, -1, :])  # Use the last hidden state
        output = self.fc(lstm_out)

        return output
        # return output

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


# Define the model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# ---
train_data = torch.load("test_data.pt")
features = []
labels=[]
for item in train_data:
    feature, label = item 
    features.append(feature)
    labels.append(label)

train_features_tensor = torch.stack(features)
train_labels = torch.tensor(labels)  
val_data = torch.load("val_data.pt")

for item in val_data:
    feature, label = item  # Unpack each tuple
    features.append(feature)
    labels.append(label)

# Stack features and labels into tensors
val_features_tensor = torch.stack(features)  # Combine 4D feature tensors
val_labels = torch.tensor(labels)

all_features = torch.cat((train_features_tensor, val_features_tensor), dim=0)
all_labels = torch.cat((train_labels, val_labels), dim=0)

# Set random seed for reproducibility
torch.manual_seed(42)

total_size = all_features.size(0)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size  # To ensure all samples are used

indices = torch.randperm(total_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]
train_features = all_features[train_indices]
train_labels = all_labels[train_indices]

val_features = all_features[val_indices]
val_labels = all_labels[val_indices]

test_features = all_features[test_indices]
test_labels = all_labels[test_indices]


print(test_features[0])
# Train and evaluate
train_loader = DataLoader(PreprocessedVideoDataset(train_features, train_labels), batch_size=10, shuffle=True)
val_loader = DataLoader(PreprocessedVideoDataset(val_features, val_labels), batch_size=10, shuffle=False)

trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)