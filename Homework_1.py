import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# =====================================================================
# 1. DEFINE LENET-5 ARCHITECTURES
# =====================================================================

class LeNet5_Baseline(nn.Module):
    """LeNet-5 without any regularization"""

    def __init__(self):
        super(LeNet5_Baseline, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14x14 -> 10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 10x10 -> 5x5

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_Dropout(nn.Module):
    """LeNet-5 with Dropout on hidden layers"""

    def __init__(self, dropout_rate=0.5):
        super(LeNet5_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first FC layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after second FC layer
        x = self.fc3(x)
        return x


class LeNet5_BatchNorm(nn.Module):
    """LeNet-5 with Batch Normalization"""

    def __init__(self):
        super(LeNet5_BatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)  # Batch norm after first conv
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)  # Batch norm after second conv
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)  # Batch norm after first FC
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)  # Batch norm after second FC
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x


# =====================================================================
# 2. LOAD FASHION-MNIST DATASET
# =====================================================================

# Transform: convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Download and load training data
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Download and load test data
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {len(train_dataset)}')
print(f'Test samples: {len(test_dataset)}')


# =====================================================================
# 3. TRAINING AND EVALUATION FUNCTIONS
# =====================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, data_loader, device):
    """Evaluate the model on a dataset"""
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def train_model(model, train_loader, test_loader, epochs, lr, weight_decay=0, model_name="Model"):
    """Complete training loop with tracking"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_accuracies = []
    test_accuracies = []

    print(f'\nTraining {model_name}...')
    print('-' * 60)

    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on test set
        test_acc = evaluate(model, test_loader, device)

        # Store accuracies
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

    print(f'Final Train Accuracy: {train_accuracies[-1]:.2f}%')
    print(f'Final Test Accuracy: {test_accuracies[-1]:.2f}%')

    return train_accuracies, test_accuracies


# =====================================================================
# 4. TRAIN ALL MODELS
# =====================================================================

epochs = 20
learning_rate = 0.001

# Dictionary to store results
results = {}

# 1. Baseline (No Regularization)
print('\n' + '=' * 60)
print('BASELINE MODEL (No Regularization)')
print('=' * 60)
model_baseline = LeNet5_Baseline()
train_acc_baseline, test_acc_baseline = train_model(
    model_baseline, train_loader, test_loader, epochs, learning_rate,
    weight_decay=0, model_name="Baseline"
)
results['Baseline'] = {
    'train': train_acc_baseline,
    'test': test_acc_baseline
}

# 2. Dropout
print('\n' + '=' * 60)
print('DROPOUT MODEL')
print('=' * 60)
model_dropout = LeNet5_Dropout(dropout_rate=0.5)
train_acc_dropout, test_acc_dropout = train_model(
    model_dropout, train_loader, test_loader, epochs, learning_rate,
    weight_decay=0, model_name="Dropout"
)
results['Dropout'] = {
    'train': train_acc_dropout,
    'test': test_acc_dropout
}

# 3. Weight Decay (L2 Regularization)
print('\n' + '=' * 60)
print('WEIGHT DECAY MODEL')
print('=' * 60)
model_weightdecay = LeNet5_Baseline()  # Same architecture as baseline
train_acc_wd, test_acc_wd = train_model(
    model_weightdecay, train_loader, test_loader, epochs, learning_rate,
    weight_decay=0.001, model_name="Weight Decay"
)
results['Weight Decay'] = {
    'train': train_acc_wd,
    'test': test_acc_wd
}

# 4. Batch Normalization
print('\n' + '=' * 60)
print('BATCH NORMALIZATION MODEL')
print('=' * 60)
model_batchnorm = LeNet5_BatchNorm()
train_acc_bn, test_acc_bn = train_model(
    model_batchnorm, train_loader, test_loader, epochs, learning_rate,
    weight_decay=0, model_name="Batch Normalization"
)
results['Batch Normalization'] = {
    'train': train_acc_bn,
    'test': test_acc_bn
}

# =====================================================================
# 5. PLOT CONVERGENCE GRAPHS
# =====================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('LeNet-5 Convergence Graphs on Fashion-MNIST', fontsize=16, fontweight='bold')

epochs_range = range(1, epochs + 1)

# Plot for each technique
techniques = ['Baseline', 'Dropout', 'Weight Decay', 'Batch Normalization']
for idx, technique in enumerate(techniques):
    # Train accuracy plot
    axes[0, idx].plot(epochs_range, results[technique]['train'], 'b-', linewidth=2)
    axes[0, idx].set_title(f'{technique}\nTrain Accuracy', fontweight='bold')
    axes[0, idx].set_xlabel('Epoch')
    axes[0, idx].set_ylabel('Accuracy (%)')
    axes[0, idx].grid(True, alpha=0.3)
    axes[0, idx].set_ylim([70, 100])

    # Test accuracy plot
    axes[1, idx].plot(epochs_range, results[technique]['test'], 'r-', linewidth=2)
    axes[1, idx].set_title(f'{technique}\nTest Accuracy', fontweight='bold')
    axes[1, idx].set_xlabel('Epoch')
    axes[1, idx].set_ylabel('Accuracy (%)')
    axes[1, idx].grid(True, alpha=0.3)
    axes[1, idx].set_ylim([70, 100])

plt.tight_layout()
plt.savefig('convergence_graphs.png', dpi=300, bbox_inches='tight')
plt.show()

print('\nConvergence graphs saved as "convergence_graphs.png"')

# =====================================================================
# 6. CREATE RESULTS TABLE
# =====================================================================

print('\n' + '=' * 60)
print('FINAL RESULTS TABLE')
print('=' * 60)
print(f'{"Technique":<25} {"Train Accuracy":<20} {"Test Accuracy":<20}')
print('-' * 60)

for technique in techniques:
    train_final = results[technique]['train'][-1]
    test_final = results[technique]['test'][-1]
    print(f'{technique:<25} {train_final:>18.2f}% {test_final:>19.2f}%')

print('=' * 60)

# =====================================================================
# 7. CONCLUSIONS
# =====================================================================

print('\n' + '=' * 60)
print('CONCLUSIONS')
print('=' * 60)

# Find best test accuracy
best_technique = max(techniques, key=lambda t: results[t]['test'][-1])
best_test_acc = results[best_technique]['test'][-1]

print(f'\n1. BEST PERFORMER: {best_technique} achieved the highest test accuracy')
print(f'   of {best_test_acc:.2f}%\n')

# Analyze overfitting
print('2. OVERFITTING ANALYSIS:')
for technique in techniques:
    train_final = results[technique]['train'][-1]
    test_final = results[technique]['test'][-1]
    gap = train_final - test_final
    print(f'   {technique}: Train-Test Gap = {gap:.2f}%')

print('\n3. REGULARIZATION EFFECTS:')
baseline_gap = results['Baseline']['train'][-1] - results['Baseline']['test'][-1]
print(f'   - Baseline gap: {baseline_gap:.2f}%')

for technique in ['Dropout', 'Weight Decay', 'Batch Normalization']:
    gap = results[technique]['train'][-1] - results[technique]['test'][-1]
    if gap < baseline_gap:
        print(f'   - {technique} REDUCED overfitting (gap: {gap:.2f}%)')
    else:
        print(f'   - {technique} did not reduce overfitting (gap: {gap:.2f}%)')

print('\n4. GENERAL OBSERVATIONS:')
print('   - Dropout typically reduces overfitting by randomly dropping neurons')
print('   - Weight Decay (L2) penalizes large weights, promoting simpler models')
print('   - Batch Normalization normalizes activations, often improving convergence')
print('   - The effectiveness of each technique depends on the specific dataset')
print('=' * 60)