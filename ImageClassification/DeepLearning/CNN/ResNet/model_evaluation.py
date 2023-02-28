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
from model import resnet34

loss = nn.CrossEntropyLoss()

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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc(fpr, tpr):
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
    plt.show()

def img_read(img_path, evalType):
    img = Image.open(img_path)
    if evalType == 'GoogleNet':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif evalType == 'ResNet':
        transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = Variable(torch.unsqueeze(transform(img), dim=0).float(), requires_grad=False)
    return img


def Precdict(img, model):
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img)).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    return predict_cla

def evaluate(txt_path, model):
    f = open(txt_path, 'r')
    lines = f.readlines()
    length = len(lines)
    pre_res = [-1]*length
    ground_truth = [-1]*length
    # fpr = [-1]*length
    # tpr = [-1] * length
    i = 0

    # 保存预测结果和真值
    for line in lines:
        img_path, label = line.split()
        res = Precdict(img_read(img_path, 'GoogleNet'), model)
        pre_res[i] = res.item()
        ground_truth[i] = int(label)

        i += 1

    print('*'*20, '\t以下显示正样本(bny)的统计指标值\t', '*'*20)
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

    # 汇总
    print('*'*20, '\t以下显示正负样本的统计指标值\t', '*'*20)
    print(classification_report(y_true=ground_truth, y_pred=pre_res))

    # ROC curve
    fpr, tpr, threshold = roc_curve(y_true=ground_truth, y_score=pre_res)
    plot_roc(fpr, tpr)


    # confusion matrix
    cnf_matrix = confusion_matrix(ground_truth, pre_res)
    np.set_printoptions(precision=4)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['infected(0)', 'uninfected(1)'], normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


if __name__ == "__main__":
    # create model
    model = resnet34(num_classes=2)

    # load model weights
    weights_path = "./resNet34.pth"
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path),
                                                          strict=False)

    evaluate('label.txt', model)