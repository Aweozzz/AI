from torch.utils.data import Dataset
import cv2
# import numpy as np

# pytorch数据集接口模板
'''
class MyDataSet(Dataset):
    def __init__(self):
        self.sample_list = ...

    def __getitem__(self, index):
        x = ...
        y = ...
        return x, y

    def __len__(self):
        return len(self.sample_list)
'''

class MyDataSet(Dataset):
    def __init__(self, txt_path=None, transform=None ):
        """
        通过txt文件实现图片路径和label的映射
        这里默认每行txt文本形式为：图片路径 + 空格 + label
        """

        self.transform = transform
        self.sample_list = list()

        f = open(txt_path, 'r')
        lines = f.readlines()

        for line in lines:
            self.sample_list.append(line.strip())

        # 记得关闭文件, 否则内容不会保存
        f.close()

    def __getitem__(self, index):
        item = self.sample_list[index]
        img = cv2.imread(item.split()[0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        # img = np.array([img])
        if self.transform is not None:
            img = self.transform(img)
        label = int(item.split()[-1])
        return img, label

    def __len__(self):
        return len(self.sample_list)




