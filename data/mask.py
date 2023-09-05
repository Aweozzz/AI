# -*- coding: utf-8 -*-

# @Time : 2023/6/20 12:35

# @Author : Aweo
# @File : mask.py

import cv2
import torch
from torchvision.models import googlenet
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import os
import torch.nn as nn
import numpy as np

def random_mask(image, mask_size, blur_sigma, save_flag=False):
    """
    mask_size 用于确定掩码模板的大小，模板形状为矩形

    blur_sigma 用于指定高斯模糊的程度。它作为cv2.GaussianBlur()函数的参数之一，用于控制模糊的程度。
    具体而言，模糊过程中使用的高斯核的大小是根据标准差自动计算的，标准差越大，高斯核的大小越大，从而产生更明显的模糊效果。
    图像坐标系：原点位于左上角，x轴向右延伸，y轴向下延伸
    """
    image = pil_to_cv2(image)
    img_h, img_w, _ = image.shape

    # 打乱像素位置
    shuffled_image = np.random.permutation(image.reshape(-1, 3)).reshape(image.shape)

    # 高斯模糊
    blurred_image = cv2.GaussianBlur(shuffled_image, (0, 0), blur_sigma)

    # 随机截取遮盖模板
    x = np.random.randint(0, img_w - mask_size[1])
    y = np.random.randint(0, img_h - mask_size[0])
    mask_template = blurred_image[y:y+mask_size[0], x:x+mask_size[1]]

    # 创建遮盖图像
    masked_image = np.copy(image)
    masked_image[y:y+mask_size[0], x:x+mask_size[1]] = mask_template


    return Image.fromarray(masked_image), x, y

def pil_to_cv2(image):
    image = image.convert("RGB")  # 将图像转换为 RGB 通道顺序
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def make_maskAndSave(img_path, predict, masked_size, sigma, rand_times, threshold, path=""):
    """
        使用固定的模板形状进行遮盖，然后对遮盖的位置和对应每个类的概率进行保存,并且保存遮盖后图片
    """
    txt_root = path + '/txt_' + str(masked_size[0]) + 'x' + str(masked_size[1]) + '_' + str(sigma) + '/'
    maskedImg_root = path + '/masked_img/' + str(masked_size[0]) + 'x' + str(masked_size[1]) + '_' + str(sigma) \
                     + '_' + (threshold) + '/'
    if not os.path.exists(txt_root):
        os.makedirs(txt_root)
    txt_path = os.path.join(txt_root, os.path.splitext(os.path.basename(img_path))[0])

    if not os.path.exists(path + '/masked_img/'):
        os.makedirs(path + '/masked_img/')
    if not os.path.exists(maskedImg_root):
        os.makedirs(maskedImg_root)

    f = open(txt_path+'.txt', 'a')
    img = Image.open(img_path)
    initial_res = predict(model, device, data_transforms, img)
    f.write(str(initial_res[0].item()) + ' ' + str(initial_res[1].item()) + ' 0' + ' ' + '0' + '\n')
    for time in range(1, rand_times+1):
        masked_image, mask_x, mask_y = random_mask(img, mask_size=masked_size, blur_sigma=sigma)
        res = predict(model, device, data_transforms, masked_image)
        if abs(res[0] - initial_res[0]) > threshold:
            f.write(str(res[0].item()) + ' ' + str(res[1].item()) + ' ' + str(mask_x) + ' ' + str(mask_y) + ' ' + '\n')
        # print(path + 'masked_img' + os.path.splitext(os.path.basename(img_path))[0] + '_' + str(time) + '.jpg')
        #     masked_image.save(maskedImg_root + os.path.splitext(os.path.basename(img_path))[0] + '_' + str(time) + '.jpg')
            masked_imageCrop = img.crop((mask_x, mask_y, mask_x+masked_size[0], mask_y+masked_size[1]))
            masked_imageCrop.save(maskedImg_root + os.path.splitext(os.path.basename(img_path))[0] + '_' + str(time)
                                   + '.tif')
    f.close()




def test_mask(image, predict_func, initial_res):
    pass

def predict(model, device, data_transforms, img):
    model.eval()
    with torch.no_grad():
        img = Variable(torch.unsqueeze(data_transforms(img), dim=0).float(), requires_grad=False)
        output = torch.squeeze(model(img.to(device))).cpu()
        result = torch.softmax(output, dim=0)
    return result



if __name__ == '__main__':
    img_root = 'Cell_20_pytorch/test/'
    masked_size = [(40, 40), (50, 50), (60, 60)]
    sigma = 20
    random_times = 50
    threshold = [0.5, 0.7, 0.9]
    work_path = "20230901"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = googlenet(pretrained=False, aux_logits=True)
    fc_futures = model.fc.in_features
    model.fc = nn.Linear(fc_futures, 2)
    weights_path = '../ImageClassification/DeepLearning/CNN/models/0901Rmsproplr1e-4Focal/GoogleNet_batch_size128_100e_lr0.0001rms.pth'
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    labels = os.listdir(img_root)
    for mk in masked_size:
        for th in threshold:
            for label in labels:
                label_path = os.path.join(img_root, label)
                img_names = os.listdir(label_path)
                for img_name in img_names:
                    img_path = os.path.join(label_path, img_name)
                    make_maskAndSave(img_path, predict, mk, sigma, random_times, threshold=th, path=work_path)

    print("done")