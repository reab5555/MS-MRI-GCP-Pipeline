import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, get_linear_schedule_with_warmup
import gcsfs
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='Python-code-running', location='europe-west1')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seed for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# Set the path to the main GCS bucket containing the class folders
gcs_bucket = 'main_il/MS_MRI_2CLASS/2C'
folders = ['NON-MS', 'MS']
gcs_folders = [os.path.join(gcs_bucket, folder) for folder in folders]

n_folds = 6
batch_size = 8
lr = 1e-5
weight_decay = 0.01
patience = 1

# Set up Google Cloud Storage File System
fs = gcsfs.GCSFileSystem()

def balance_dataset(images, labels):
    class_counts = {}
    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    print("Class counts before balancing:", class_counts)
    min_count = min(class_counts.values())
    print(f"Balancing to {min_count} samples per class")

    balanced_images = []
    balanced_labels = []
    class_counters = {label: 0 for label in class_counts}

    for image, label in zip(images, labels):
        if class_counters[label] < min_count:
            balanced_images.append(image)
            balanced_labels.append(label)
            class_counters[label] += 1

    return balanced_images, balanced_labels

class CustomImageDataset(Dataset):
    def __init__(self, gcs_folders, processor):
        self.processor = processor
        self.images = []
        self.labels = []

        for i, folder in enumerate(gcs_folders):
            print(f"Processing folder: {folder}")
            folder_images = fs.ls(folder)
            print(f"  Found {len(folder_images)} images")
            self.images.extend(folder_images)
            self.labels.extend([i] * len(folder_images))

        unique_labels = set(self.labels)
        print(f"Unique labels before balancing: {unique_labels}")
        print(f"Total images before balancing: {len(self.images)}")

        # Balance the dataset
        self.images, self.labels = balance_dataset(self.images, self.labels)

        print(f"After balancing:")
        unique_labels = set(self.labels)
        for label in unique_labels:
            count = self.labels.count(label)
            print(f"  Class {label}: {count} images")

        print(f"Total images after balancing: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Open image from GCS
        with fs.open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        image = image.resize((384, 384))
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return pixel_values, label

# Load the ViT model and image processor
model_name = "google/vit-base-patch16-384"
image_processor = ViTImageProcessor.from_pretrained(model_name)

class CustomViTModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomViTModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(model_name)

        # Replace the classifier
        num_features = self.vit.config.hidden_size
        self.vit.classifier = nn.Identity()  # Remove the original classifier

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Single output for binary classification
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        x = outputs.logits
        x = F.adaptive_avg_pool2d(x.unsqueeze(-1).unsqueeze(-1), (1, 1)).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze()

def train_or_evaluate(model, loader, optimizer, criterion, device, is_training, scheduler=None):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(is_training):
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, all_labels, all_preds, f1

# Create full dataset
full_dataset = CustomImageDataset(gcs_folders, image_processor)

# Get class names
class_names = [folder.split('/')[-1] for folder in gcs_folders]
print(f"Class names: {class_names}")

# Prepare for cross-validation
n_splits = n_folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

# Cross-validation loop
fold_results = []
all_val_labels = []
all_val_preds = []
all_train_losses = []
all_val_losses = []

for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset.images, full_dataset.labels), 1):
    print(f"Fold {fold}/{n_splits}")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler)

    model = CustomViTModel()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    best_val_loss = float('inf')
    patience_counter = 0

    fold_train_losses = []
    fold_val_losses = []

    for epoch in range(num_epochs):
        train_loss, _, _, _ = train_or_evaluate(model, train_loader, optimizer, criterion, device, is_training=True, scheduler=scheduler)
        val_loss, val_labels, val_preds, val_f1 = train_or_evaluate(model, val_loader, optimizer, criterion, device, is_training=False)

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)

        # Calculate precision, recall, and f1-score
        precision = precision_score(val_labels, val_preds, average='weighted')
        recall = recall_score(val_labels, val_preds, average='weighted')
        f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Log metrics to Vertex AI
        aiplatform.log_metric('train_loss', train_loss)
        aiplatform.log_metric('val_loss', val_loss)
        aiplatform.log_metric('precision', precision)
        aiplatform.log_metric('recall', recall)
        aiplatform.log_metric('f1_score', f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Store losses for this fold
    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Final evaluation
    _, val_labels, val_preds, val_f1 = train_or_evaluate(model, val_loader, optimizer, criterion, device, is_training=False)
    report = classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
    fold_results.append({
        'accuracy': (np.array(val_labels) == np.array(val_preds)).mean(),
        'f1': val_f1,
        'report': classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
    })
    all_val_labels.extend(val_labels)
    all_val_preds.extend(val_preds)

    print(f"Fold {fold} results:")
    print(classification_report(val_labels, val_preds, target_names=class_names))

    # You can also log fold-specific metrics
    aiplatform.log_metric(f'fold_{fold}_accuracy', (np.array(val_labels) == np.array(val_preds)).mean())
    aiplatform.log_metric(f'fold_{fold}_f1_score', val_f1)
    print(f"Fold {fold} results:\n{report}")

# Calculate average metrics across folds
avg_results = {
    'accuracy': np.mean([fold['accuracy'] for fold in fold_results]),
    'f1': np.mean([fold['f1'] for fold in fold_results]),
    'report': {
        class_name: {
            metric: np.mean([fold['report'][class_name][metric] for fold in fold_results])
            for metric in ['precision', 'recall', 'f1-score']
        }
        for class_name in class_names
    }
}

print("\nAverage results across all folds:")
print(f"Accuracy: {avg_results['accuracy']:.4f}")
print(f"F1 Score: {avg_results['f1']:.4f}")
for class_name in class_names:
    print(f"\n{class_name}:")
    for metric, value in avg_results['report'][class_name].items():
        print(f"  {metric}: {value:.4f}")


# Get the current timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the final model with a timestamp
model_filename = f'final_ms_model_2classes_vit_base_384_bce_{timestamp}.pth'
torch.save(model.state_dict(), model_filename)
print(f"Final model saved as '{model_filename}'")
