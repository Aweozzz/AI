import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CNN

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    train_steps = len(train_loader)
    train_bar = tqdm(train_loader, file=sys.stdout)
    train_num = len(train_loader.dataset)
    correct = 0

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = "train loss: {:.3f}".format(running_loss / (train_steps * len(images)))

    return running_loss / train_steps


def validate(model, device, val_loader):
    model.eval()
    val_num = len(val_loader.dataset)
    correct = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    acc = correct / val_num

    return acc


def main(batch_size=32, lr=5e-5, epochs=50, num_workers=0, save_path=''):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transforms = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data_dir = "J:/destop/毕设/data/data"
    assert os.path.exists(data_dir), "{} path does not exist.".format(data_dir)

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True).to(device)
    model = CNN()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_acc = validate(model, device, val_loader)
        print("epoch [{}/{}], train loss: {:.4f}, val accuracy: {:.4f}".format(epoch+1, epochs, train_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)





if __name__ == '__main__':
    main(save_path="article_net.pth")