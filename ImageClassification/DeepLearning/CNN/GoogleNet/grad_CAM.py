import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
from PIL import Image, ImageOps

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.feature_map = None
        self.gradient = None

        def save_feature_map(module, input, output):
            self.feature_map = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        self.target_layer.register_forward_hook(save_feature_map)
        self.target_layer.register_backward_hook(save_gradient)

    def forward(self, x):
        return self.model(x)

    def __call__(self, x, index=None):
        output = self.forward(x)
        if index is None:
            index = output.argmax(dim=1)

        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, index.view(-1, 1), 1)

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        weight = self.gradient.mean(dim=(-2, -1), keepdim=True)
        cam = (weight * self.feature_map).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        return cam.squeeze(0).cpu().numpy()

def main():
    # 数据预处理
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载预训练的 GoogLeNet 模型
    model = torchvision.models.googlenet(pretrained=False)

    # 替换分类层以适应二分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    weights_path = r"E:\GangLiu\AI\ImageClassification\DeepLearning\CNN\models\0419random_crop\GoogleNet_batch_size128_50e_lr1e-05_adam.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)

    # 使用最后一个卷积层作为 Grad-CAM 的目标层
    target_layer = model.inception5b.branch4[1].conv

    # 实例化 GradCAM 对象
    grad_cam = GradCAM(model, target_layer)

    # 读取一张图像
    img_path = 'C:/Users/Administrator/Desktop/fig/random_crop_-1.jpg'
    img = Image.open(img_path).convert('RGB')

    # 数据预处理
    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)

    # 生成 Grad-CAM
    cam = grad_cam(input_tensor)

    # 可视化
    cam = cam.squeeze(0)
    cam = Image.fromarray(np.uint8(255 * cam)).resize(img.size, Image.BICUBIC)
    heatmap = ImageOps.colorize(cam, "blue", "red")
    result = Image.blend(img, heatmap, alpha = 0.5)
    # 保存并显示结果
    result.save('C:/Users/Administrator/Desktop/fig/random_crop_-1_GradCAM.jpg')
    result.show()

if __name__ == "__main__":
    main()