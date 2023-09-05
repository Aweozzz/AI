import os
import sys
from math import exp
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import classification_report,confusion_matrix
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torchvision.models import googlenet
from torchvision.models import resnet34
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).type(inputs.dtype)

        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        at = self.alpha.gather(0, targets.data.view(-1).long())

        F_loss = at * (1 - pt) ** self.gamma * F.cross_entropy(inputs, targets, reduction='none')
        return F_loss.mean()
loss = nn.CrossEntropyLoss()
# loss = WeightedFocalLoss()
def sigmoid(x):
    return 1/(1+exp(-x))

def plot_confusion_matrix(cm, classes,
                          normalize=False, #if true all values in confusion matrix is between 0 and 1
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True test')
    plt.xlabel('Predicted test')

def plot_roc(fpr, tpr, model_name, path):
    plt.figure(figsize=(15, 10), dpi=300)
    lw = 1
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area = %0.4f)' % auc(fpr, tpr))

    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path, model_name+'_roc.png'))
    print("roc png saved")
    # plt.show()

def img_read(img_path, evalType):
    img = Image.open(img_path).convert('RGB')
    # img = Image.open(img_path)
    # print(img.mode)
    if evalType == 'googlenet':
        transform = transforms.Compose(
            [
            # transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Resize((224, 224)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    elif evalType == 'resnet-34':
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    elif evalType == 'articleCNN':
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    elif evalType == 'resnet-34Gray':
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    elif evalType == 'googlenetGray':
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    img = Variable(torch.unsqueeze(transform(img), dim=0).float(), requires_grad=False)
    return img


def Precdict(img, model, device):
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    return predict_cla, predict

def evaluate(txt_path, model, model_name, path, device):
    f = open(txt_path, 'r')
    lines = f.readlines()
    length = len(lines)
    pre_res = [-1]*length
    ground_truth = [-1]*length
    pre_p = [-1]*length
    i = 0

    # 保存预测结果和真值
    for line in lines:
        img_path, label = line.split()
        res, res_p = Precdict(img_read(img_path, model_name), model, device)
        pre_res[i] = res.item()
        pre_p[i] = res_p[1]
        ground_truth[i] = int(label)
        if pre_res[i] == -1:
            pre_res[i] = 0
        if ground_truth[i] == -1:
            ground_truth[i] = 0
        i += 1

    ff = open(path+'/'+model_name+'report.txt', 'w')
    print('*'*20, '\t以下显示统计指标值\t', '*'*20)
    # accuracy
    ACC = accuracy_score(y_true=ground_truth, y_pred=pre_res)
    print('accuracy_score:', ACC)

    # precision
    PRE = precision_score(y_true=ground_truth, y_pred=pre_res)
    print('precision_score:', PRE)

    # F1-score
    F1 = f1_score(y_true=ground_truth, y_pred=pre_res)
    print('f1_score:', F1)

    # recall
    REC = recall_score(y_true=ground_truth, y_pred=pre_res)
    print('recall_score:', REC,end='\n\n')

    #txt record
    ff.write('accuracy_score:'+str(ACC)+'\n'+
            'precision_score:'+str(PRE)+'\n'+
            'f1_score:'+str(F1)+'\n'+
            'recall_score:'+str(REC)+'\n')
    ff.close()

    # 汇总
    print('*'*20, '\t以下显示正负样本的统计指标值\t', '*'*20)
    print(classification_report(y_true=ground_truth, y_pred=pre_res))

    # ROC curve
    fpr, tpr, threshold = roc_curve(y_true=ground_truth, y_score=pre_p)
    plot_roc(fpr, tpr, model_name, path)


    # confusion matrix
    cnf_matrix = confusion_matrix(ground_truth, pre_res)
    np.set_printoptions(precision=4)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['-1', '1'], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(path+'/'+model_name+'_cm.png')

def run(work_path=''):
    model_path_list = []
    for file_name in os.listdir(work_path):
        if file_name.endswith('pth'):
            model_path_list.append(os.path.join(work_path, file_name))

    for path in model_path_list:
        if path.find('GoogleNet'):
            model = googlenet(pretrained=True, aux_logits=True)
            # fc_features = model.fc.in_features
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(path, map_location=device)
            model.fc = nn.Linear(checkpoint['fc.weight'].shape[1], 2)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

            test_datapath = '../../../data/Cell_20_pytorch/test'
            os.system('python txt_generate.py' + ' ' + test_datapath)
            model.to(device)
            evaluate('label.txt', model, model_name='googlenet',
                     path=os.path.dirname(path), device=device)

        if path.find('ResNet'):
            model = resnet34(num_classes=2)
            model.cuda()
            # fc_features = model.fc.in_features
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(path, map_location=device)
            model.fc = nn.Linear(checkpoint['fc.weight'].shape[1], 2)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

            test_datapath = '../../../data/Cell_20_pytorch/test'
            os.system('python txt_generate.py' + ' ' + test_datapath)
            model.to(device)
            evaluate('label.txt', model, model_name='resnet-34',
                     path=os.path.dirname(path), device=device)



if __name__ == "__main__":
    """
        In order to evaluate your model,you need to follow 3 steps:
        step1:choose your model type
        step2:choose your model saving path
        step3:make sure you test image labels path is correct
    """
    args = sys.argv[1:]
    run(work_path=args[0])