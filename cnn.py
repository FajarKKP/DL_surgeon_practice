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
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_initial, shuffle = True, num_workers= 0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size= batch_size_initial, shuffle = False, num_workers= 0)

# Models
class Cnn1(nn.Module):
    def __init__(self, c1,c2,c3,c4,c5,c6):
        super(Cnn1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c2,c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c3,c4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c5,c6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(c6, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class Cnn2(nn.Module):
    def __init__(self, c1,c2,c3,c4,c5,c6):
        super(Cnn2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2),

            nn.Conv2d(c2,c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c3,c4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c5, c5, kernel_size=3, stride = 2, padding=1),

            nn.Conv2d(c5,c6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(c6, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class Cnn3(nn.Module):
    def __init__(self, c1,c2,c3,c4,c5,c6):
        super(Cnn3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c2, kernel_size=3, stride = 2, padding=1),

            nn.Conv2d(c2,c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c3,c4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c4, c5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c5, c5, kernel_size=3, stride = 2, padding=1),

            nn.Conv2d(c5,c6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(c6, 10)

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
if __name__ == "__main__":
    # print("=== BASELINE MODEL ===")
    # baseline1 = Cnn1(32, 64, 64, 128, 128, 256).to(device)
    # train_model(baseline1)
    # evaluate_model(baseline1, "Baseline1")

    print("=== BASELINE MODEL ===")
    baseline2 = Cnn2(32, 64, 64, 128, 128, 256).to(device)
    train_model(baseline2)
    evaluate_model(baseline2, "Baseline2")

    print("=== BASELINE MODEL ===")
    baseline3 = Cnn3(32, 64, 64, 128, 128, 256).to(device)
    train_model(baseline3)
    evaluate_model(baseline3, "Baseline3")

    # print("\n=== DOUBLE WIDTH MODEL ===")
    # double_width = Cnn(64, 128, 256).to(device)
    # train_model(double_width)
    # evaluate_model(double_width, "Double Width")

    # print("=== ANOTHER MODEL ===")
    # another_baseline = Cnn(32, 32, 64, 64, 128, 256).to(device)
    # train_model(another_baseline)
    # evaluate_model(another_baseline, "Another Baseline")


