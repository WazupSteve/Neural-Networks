# main.py
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os

# VGG16 Model Definition
class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Trainer:
    def __init__(self, model, data_dir='./data', batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.005)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_loader, self.valid_loader = self._get_data_loaders()
        self.test_loader = self._get_test_loader()

    def _get_data_loaders(self, valid_size=0.1, random_seed=42):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])
        
        train_dataset = datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=transform)
        
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler)
        
        return train_loader, valid_loader

    def _get_test_loader(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])
        test_dataset = datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, num_epochs=20):
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
                
                del images, labels, outputs
                torch.cuda.empty_cache()

            # Save model checkpoint
            torch.save(self.model.state_dict(), f'./model_checkpoint_epoch_{epoch + 1}.pth')
            print(f"Model checkpoint saved at epoch {epoch + 1}")

            # Validation
            val_accuracy = self.validate()
            print(f"Validation accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")

    def validate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
                torch.cuda.empty_cache()
        return 100 * correct / total

    def test(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
                torch.cuda.empty_cache()
        accuracy = 100 * correct / total
        print(f"Test accuracy: {accuracy:.2f}%")
        return accuracy

def main():
    model = VGG16(num_classes=100)
    trainer = Trainer(model)
    trainer.train()
    trainer.test()

if __name__ == "__main__":
    main()