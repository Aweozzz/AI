import os
import json
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.models import googlenet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "E:/GangLiu/AI/data/20230811Test/test/1/ZYCLPAAD02_A1_crop_1201.tif"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)


    # create model
    # model = GoogLeNet(num_classes=2, aux_logits=False).to(device)
    # model = googlenet(pretrained=False, aux_logits=True)
    # fc_futures = model.fc.in_features
    # model.fc = nn.Linear(fc_futures, 2)
    model = googlenet(pretrained=False, aux_logits=True)
    fc_futures = model.fc.in_features
    model.fc = nn.Linear(fc_futures, 2)
    model.to(device)
    # load model weights
    weights_path = "../models/LKYtest/GoogleNet_batch_size128_200e_lr1e-05_adam_96.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    class_indict = {'0':'naiyao', '1':'bunaiyao'}
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(len(predict))
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
