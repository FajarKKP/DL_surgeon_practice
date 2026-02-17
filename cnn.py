import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from ptflops import get_model_complexity_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size_initial = 128
epochs_initial = 15
lr_initial = 0.001

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding  = 4),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_initial, shuffle = True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size_initial, shuffle = False, num_workers= 2)

# Model
class Cnn(nn.Module):
    def __init__(self, c1,c2,c3):
        super(Cnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2,c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(c3, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
    
# Train
def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_initial)

    for epoch in range(epochs_initial):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total

        print(f"Epoch {epoch+1}/{epochs_initial} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# ----------------------------
# Evaluation Metrics
# ----------------------------
def evaluate_model(model, name):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{name} Parameters: {total_params:,}")

    # FLOPs
    with torch.cuda.device(0) if torch.cuda.is_available() else torch.device("cpu"):
        macs, params = get_model_complexity_info(
            model, (3, 32, 32), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        print(f"{name} FLOPs: {macs}")

    # Inference time
    dummy = torch.randn(1, 3, 32, 32).to(device)
    start = time.time()
    for _ in range(100):
        _ = model(dummy)
    end = time.time()

    print(f"{name} Avg Inference Time (100 runs): {(end - start)/100:.6f} sec")

# ----------------------------
# Run Baseline
# ----------------------------
print("=== BASELINE MODEL ===")
baseline = Cnn(32, 64, 128).to(device)
train_model(baseline)
evaluate_model(baseline, "Baseline")

# ----------------------------
# Run Double Width
# ----------------------------
print("\n=== DOUBLE WIDTH MODEL ===")
double_width = Cnn(64, 128, 256).to(device)
train_model(double_width)
evaluate_model(double_width, "Double Width")




