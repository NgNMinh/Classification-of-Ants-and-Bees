from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os


def train():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder('Datasets/hymenoptera_data/train', train_transforms)
    val_dataset = ImageFolder('Datasets/hymenoptera_data/val', val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    model.to(device)

    optim = SGD(model.parameters(), 0.001, 0.9)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 25
    early_stopping = 0
    best_accuracy = -1

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_dataloader)
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)

            loss.backward()
            optim.step()
            optim.zero_grad()
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, epochs, loss))

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs)
                all_preds.extend(torch.argmax(preds, dim=1).tolist())
                all_labels.extend(labels.tolist())
        accuracy = accuracy_score(all_labels, all_preds)
        print(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(model_path, 'best.pt'))
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping == 5:
                exit(0)


if __name__ == '__main__':
    train()
