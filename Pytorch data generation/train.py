import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import get_mydata
import os
from torch.utils.tensorboard import SummaryWriter


def load_data(classes, trainTXT_path, testTXT_path):
    # Normolize the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = get_mydata.MyDataSet(txt_path=trainTXT_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    test_set = get_mydata.MyDataSet(txt_path=testTXT_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    classes = classes
    return train_loader, test_loader


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    writer.add_scalar('train_loss', loss, epoch)


def test_loop(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    global Accuracy
    Accuracy = (100 * correct)
    print(f"Test Error: \n Accuracy: {Accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalar('test_loss', test_loss, epoch)


if __name__ == '__main__':
    #输入训练参数
    batch_size = 32
    num_workers = 2
    Epoch = 50
    classes = ['0', '1']
    model_path = ''
    trainTXT_path = ''
    testTXT_path = ''

    # choose your net
    net = ''

    log_path = 'log/' + model_path.split('.')[0]

    cp = True  # 是否打开检查点
    if cp:
        checkpoint_path = 'checkpoint/' + model_path.split('.')[0]
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        f = open(checkpoint_path+'/logs.txt', 'a')

    # TensorBoard setting
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(log_dir=log_path)
    writer.add_graph(net, torch.Tensor(batch_size, 3, 32, 32))

    print('*' * 30)
    print('\t\tloading the data')
    print('*' * 30)

    # load the data
    train_loader, test_loader = load_data(classes, trainTXT_path, testTXT_path)

    # check the gpu device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    net.to(device)

    # 定义一个优化器 和 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    # 训练
    for t in range(Epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, net, criterion, optimizer, t)
        test_loop(test_loader, net, criterion, t)
        if cp:
            # model_path = os.path.join(checkpoint_path + 'Epoch:%d Accuracy:(%.2f).pth' % (t, Accuracy))
            model_path = os.path.join(checkpoint_path + '\%d.pth' % t)
            f.write('Epoch:%d Accuracy:%f\n' % (t, Accuracy))
            torch.save(net.state_dict(), model_path)

    writer.close()
    f.close()
    print("Done!")
