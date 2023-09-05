import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import googlenet
from torchvision.models.googlenet import GoogLeNet
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetClassifier
import numpy as np

# 设置数据转换
transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Resize((224, 224)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载训练和验证数据集
train_dataset = datasets.ImageFolder(root='E:\GangLiu\AI\data/Cell_20_pytorch/train', transform=transform)
valid_dataset = datasets.ImageFolder(root='E:\GangLiu\AI\data/Cell_20_pytorch/val', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

class CustomGoogLeNet(GoogLeNet):
    def __init__(self, num_classes=2, aux_logits=True, transform_input=False, init_weights=True):
        super().__init__(num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input, init_weights=init_weights)

    def forward(self, x):
        x = self._transform_input(x)
        x, _, _ = self._forward(x)
        return x

def create_googlenet_instance():
    model_instance = CustomGoogLeNet(aux_logits=True)
    fc_futures = model_instance.fc.in_features
    model_instance.fc = nn.Linear(fc_futures, 2)
    return model_instance

net = NeuralNetClassifier(
    module=create_googlenet_instance(),
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    lr=0.001,
    iterator_train__shuffle=True,
    verbose=1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)


# 设置随机搜索参数
param_dist = {
    "lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "batch_size": [16, 32, 64, 128],
    "max_epochs": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "optimizer": [optim.SGD, optim.Adam, optim.RMSprop], # 添加优化器参数
}

# 创建随机搜索
random_search = RandomizedSearchCV(
    net,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=1,
    refit=False,
)

def extract_data_labels(dataloader):
    data, labels = [], []
    for images, targets in dataloader:
        data.append(images)
        labels.append(targets)

    data = torch.cat(data, 0)
    labels = torch.cat(labels, 0)
    return data, labels

# 提取数据和标签
X_train, y_train = extract_data_labels(train_loader)
X_valid, y_valid = extract_data_labels(valid_loader)

# 转换为 NumPy 数组
X_train, y_train = X_train.numpy(), y_train.numpy()
X_valid, y_valid = X_valid.numpy(), y_valid.numpy()

random_search.fit(X_train, y_train)
print("Best parameters found: ", random_search.best_params_)