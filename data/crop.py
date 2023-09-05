import PIL.Image as Image
import os
from torchvision import transforms as transforms
import pandas as pd
from torchvision.transforms.functional import crop

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
def Custom_crop(image, x, y, size=224):
    return crop(image, y-size, x-size, size, size)

def horizonflip(image):
    HF = transforms.RandomHorizontalFlip(p=1.0)
    return HF(image)


def verticalflip(image):
    VF = transforms.RandomVerticalFlip(1.0)
    return VF(image)


def judge_RD(x, y):
    # 判断是否会选取到右下角比例尺标签
    if x >= 1450 or y >= 1150:
        return False
    return True

def judge_bound(x, y, x_bound, y_bound, size = 224):
    if x-size >= 0 and x+size < x_bound and y-size >= 0 and y-size<y_bound:
        return True
    else:
        return False

if __name__ == '__main__':
    img_r = ["23_2_28/10x/1", "23_2_28/10x/-1"]
    img_type = '.jpg'
    # csv_r = [i + '_csv' for i in img_r]
    csv_r = [i + '_csv_Ncluster' for i in img_r]
    Size = 224
    # n = 200
    # prop = [0.8 * n, 0.9 * n, 1 * n]

    for img_root, csv_root in zip(img_r, csv_r):
        Train_path = "0519cluster_crop/" + "train" + '/' + img_root.split('/')[-1]
        Test_path = "0519cluster_crop/" + "test" + '/' + img_root.split('/')[-1]
        Val_path = "0519cluster_crop/" + "val" + '/' + img_root.split('/')[-1]
        if not os.path.exists(Train_path):
            os.makedirs(Train_path)
        if not os.path.exists(Val_path):
            os.makedirs(Val_path)
        if not os.path.exists(Test_path):
            os.makedirs(Test_path)

        for img_r in os.listdir(img_root):
            img_r = os.path.join(img_root, img_r)
            image = read_PIL(img_r)
            # print(image.size)  # 输出原图像的尺寸
            name = get_name(image) # 获取读到图片的不带后缀的名称
            csv_path = os.path.join(csv_root, name+".csv")
            df = pd.read_csv(csv_path)
            df = df.loc[:, ['X', 'Y']]

            n = len(df)
            prop = [0.8 * n, 0.9 * n, 1 * n]
            count = 0
            for indexs in df.index:
                count += 1
                x, y = df.iloc[indexs].values[:]
                if judge_RD(x, y) and judge_bound(x, y, x_bound=1600, y_bound=1200, size=Size):

                    crop_image = Custom_crop(image, x, y, size=Size)

                    out_name = name + '_crop_' + str(count) + img_type
                    if count <= prop[0]:
                        crop_image.save(os.path.join(Train_path, out_name))  # 按照路径保存图片
                    elif count <= prop[1]:
                        crop_image.save(os.path.join(Test_path, out_name))  # 按照路径保存图片
                    elif count <= prop[2]:
                        crop_image.save(os.path.join(Val_path, out_name))  # 按照路径保存图片
    print("done!")