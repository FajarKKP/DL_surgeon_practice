import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from ptflops import get_model_complexity_info

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Reproducibility
# ----------------------------
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 256
epochs = 15
lr = 0.001
weight_decay = 5e-4

# ----------------------------
# Transforms
# ----------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# ----------------------------
# Dataset & Loader
# ----------------------------
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ----------------------------
# CNN Model (Stride variant)
# ----------------------------
class CnnStride1(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5, c6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.Conv2d(c3, c4, 3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.Conv2d(c4, c5, 3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c5, c6, 3, padding=1),
            nn.BatchNorm2d(c6),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(c6, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CnnStride2(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5, c6,
                 c7, c8, c9, c10, c11, c12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.Conv2d(c3, c4, 3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.Conv2d(c4, c5, 3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c5, c6, 3, padding=1),
            nn.BatchNorm2d(c6),
            nn.ReLU(),

            nn.Conv2d(c7, c8, 3, padding=1),
            nn.BatchNorm2d(c8),
            nn.ReLU(),
            nn.Conv2d(c8, c9, 3, padding=1),
            nn.BatchNorm2d(c9),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c9, c10, 3, padding=1),
            nn.BatchNorm2d(c10),
            nn.ReLU(),
            nn.Conv2d(c10, c11, 3, padding=1),
            nn.BatchNorm2d(c11),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c11, c12, 3, padding=1),
            nn.BatchNorm2d(c12),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(c12, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ----------------------------
# Train Function
# ----------------------------
def train_model(model):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        model.train()
        correct = 0
        total = 0
        # running_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.abs().mean())
            optimizer.step()

            # running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        # print(("Epoch loss:", running_loss / len(trainloader)))

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_model(model, name):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{name} Parameters: {total_params:,}")

    macs, _ = get_model_complexity_info(
        model,
        (3, 32, 32),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
    print(f"{name} FLOPs: {macs}")

    dummy = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(dummy)
        end = time.time()
    print(f"{name} Avg Inference Time (100 runs): {(end - start)/100:.6f} sec")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    model1 = CnnStride1(32, 64, 64, 128, 128, 256)
    train_model(model1)
    evaluate_model(model1, "Stride Model")

    # model2 = CnnStride2(32, 32, 64, 64, 128, 128, 128, 128, 256,256, 512,512)
    # train_model(model2)
    # evaluate_model(model2, "Stride Model")