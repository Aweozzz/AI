"""
author:chen_ziying
"""
import shutil

import PIL.Image as Image
import os
from torchvision import transforms as transforms
from torchvision.transforms.functional import crop
import random


Image.MAX_IMAGE_PIXELS = 500000000

# pytorch提供的torchvision主要使用PIL的Image类进行处理，所以它数据增强函数大多数都是以PIL作为输入，并且以PIL作为输出。
# 读取图片
def read_PIL(image_path):
    image = Image.open(image_path)
    return image


# 获取读到图片的不带后缀的名称
def get_name(image):
    im_path = image.filename
    im_name = os.path.split(im_path)  # 将路径分解为路径中的文件名+扩展名，获取到的是一个数组格式，最后一个是文件名
    name = os.path.splitext(im_name[len(im_name) - 1])  # 获取不带扩展名的文件名，是数组的最后一个
    return name[0]  # arr[0]是不带扩展名的文件名，arr[1]是扩展名


# 将图片Resize
def resize_img(image, size=224):
    Resize = transforms.Resize(size)
    resize_img = Resize(image)
    return resize_img


# 裁剪
def Custom_crop(image, x, y,size = 224):
    return crop(image, y-size, x-size, size, size)

def horizonflip(image):
    HF = transforms.RandomHorizontalFlip(p=1.0)
    return HF(image)


def verticalflip(image):
    VF = transforms.RandomVerticalFlip(1.0)
    return VF(image)

def judge_bound(x, y, x_bound, y_bound, size = 224):
    if x-size >= 0 and x+size < x_bound and y-size >= 0 and y-size<y_bound:
        return True
    else:
        return False


def data_assemble(root, out_path, img_type):
    """
    将每个大图筛选后的子图进行遍历切分后混在一起（train， test， val）
    :parm root:initial path root
    :parm out_path:output path for saving sub images
    :return :None
    """
    label = os.listdir(root)
    for l in label[0:2]:
        img_label_path = os.path.join(root, l)
        target_path = os.path.join(out_path, l)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for cell_id in os.listdir(img_label_path):
            cell_id_path = os.path.join(img_label_path, cell_id)
            cell_id_path = os.path.join(cell_id_path, 'NewCrop')
            print(cell_id_path, 'processing...')
            for img_name in os.listdir(cell_id_path):
                count = 1
                img_path = os.path.join(cell_id_path, img_name)
                tif_img = read_PIL(img_path)
                width_px, height_px = tif_img.size
                i_max, j_max = height_px // 224, width_px // 224
                for i in range(0, i_max):
                    for j in range(0, j_max):
                        i_px = i * 224
                        j_px = j * 224
                        crop_img = crop(tif_img, i_px, j_px, 224, 224)
                        crop_img.save(target_path + '/' + get_name(tif_img) + '_' + str(count) + img_type)
                        count += 1
            print(cell_id_path, 'done!')
def pytorch_data_generate(root, out_path):
    """
    对分好类的文件夹中的224x224图片进行打乱后划分数据集(train, test, val)
    :parm root:224x224 images' path(include labels)
    :parm out_path:path for saving dataset
    :return None
    """
    label = os.listdir(root)
    proposion = [0.6, 0.8, 1]
    for l in label:
        # label_path_out = os.path.join(out_path, l)
        label_path_origin = os.path.join(root, l)
        train_path = os.path.join(out_path, 'train/'+l)
        test_path = os.path.join(out_path, 'test/'+l)
        val_path = os.path.join(out_path, 'val/'+l)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        img_path = [os.path.join(label_path_origin, i) for i in os.listdir(label_path_origin)]
        random.shuffle(img_path)
        n = len(img_path)
        count = 0
        for i in range(0, n):
            count += 1
            if count < n*proposion[0]:
                shutil.copy(img_path[i], train_path)
            elif count < n*proposion[1]:
                shutil.copy(img_path[i], test_path)
            else:
                shutil.copy(img_path[i], val_path)
    print('done!')
if __name__ == '__main__':
    # data_assemble(r"D:\cyh\20株细胞",out_path=r'D:\cyh\LiugTest', img_type='.tif')
    pytorch_data_generate(r'E:\GangLiu\AI\data\Cell_20_Full', out_path=r'E:\GangLiu\AI\data\Cell_20_pytorch')