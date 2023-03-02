import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.googlenet
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import googlenet
from matplotlib import pyplot as plt
import numpy as np



def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    train_steps = len(train_loader)
    train_bar = tqdm(train_loader, file=sys.stdout)
    train_num = len(train_loader.dataset)
    correct = 0
    correct2 = 0
    correct1 = 0

    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = model(images)
        loss0 = criterion(logits, labels)
        loss1 = criterion(aux_logits1, labels)
        loss2 = criterion(aux_logits2, labels)
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        pred2 = aux_logits2.argmax(dim=1, keepdim=True)
        correct2 += pred2.eq(labels.view_as(pred2)).sum().item()
        pred1 = aux_logits1.argmax(dim=1, keepdim=True)
        correct1 += pred1.eq(labels.view_as(pred1)).sum().item()

        running_loss += loss.item()

        train_bar.desc = "train loss: {:.3f}".format(running_loss / (train_steps * len(images)))

    train_acc =  correct / train_num

    return running_loss / train_steps, train_acc


def validate(model, device, val_loader, criterion):
    model.eval()
    val_step = len(val_loader)
    val_num = len(val_loader.dataset)
    correct = 0
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss += loss.item()

    val_acc = correct / val_num

    return val_loss / val_step, val_acc


def main(batch_size=32, lr=1e-5, epochs=5, num_workers=0, save_path='./googlenet.pth'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data_dir = "../../../data/mydate"
    assert os.path.exists(data_dir), "{} path does not exist.".format(data_dir)

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True).to(device)
    model = googlenet(num_classes = 2)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0

    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print("epoch [{}/{}], train loss: {:.4f}, train accuracy: {:.4f}, val loss {:.4f}, val accuracy: {:.4f}"
              .format(epoch+1, epochs, train_loss, train_acc, val_loss, val_acc))
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        val_losses.append(val_loss)
        val_acces.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)



    # 绘图代码
    plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
    plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")
    plt.plot(np.arange(len(val_losses)), val_losses, label="valid loss")
    plt.plot(np.arange(len(val_acces)), val_acces, label="valid acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    plt.title('Model accuracy&loss')
    plt.savefig("model.png")
    # plt.show()



if __name__ == '__main__':
    main()