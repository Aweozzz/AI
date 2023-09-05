import os.path
from PIL import Image
import os
from pixelmatch.contrib.PIL import pixelmatch


def clean_duplicase(images, img_size, th=0.5):
    out_imagePath = []
    out_imagePath.append(images[0])
    origin_nums = len(images)
    mark = [0] * origin_nums
    for i in  range(0, origin_nums-1):
        if mark[i] == 0:
            img_a = Image.open(images[i])
            for ii in range(i+1, origin_nums):
                if mark[ii] == 0:
                    img_b = Image.open(images[ii])
                    if pixelmatch(img_a, img_b) > img_size[0] * img_size[1] * th:
                        mark[ii] = 1
                    else:
                        img_b.close()
            img_a.close()
    for i in range(1, origin_nums):
        if mark[i] == 0:
            out_imagePath.append(images[i])

    return out_imagePath


def generate(in_path, out_path, clean_duplicase=False):
    out_path = os.path.join(out_path, in_path.split('\\')[-2])
    out_path = os.path.join(out_path, in_path.split('\\')[-1])
    # print(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sub_images = [os.path.join(in_path, path) for path in os.listdir(in_path)]
    ig = Image.open(sub_images[0])
    sub_image_width, sub_image_height = ig.width, ig.height
    ig.close()
    if clean_duplicase:
        sub_images = clean_duplicase(sub_images, (ig.width, ig.height), 0.3)
    # print(sub_images)

    # 指定大图的列数（每行自动计算）
    num_cols = 10
    target_nums = len(sub_images) // (num_cols * num_cols)
    # 创建空白大图
    id = 0
    for i in range(1, target_nums+1):
        large_image_width = num_cols * sub_image_width
        large_image_height = num_cols * sub_image_height
        large_image = Image.new('RGB', (large_image_width, large_image_height))

        for ii in range(0, num_cols*num_cols):
            sub_image = Image.open(sub_images[id])
            id += 1
            # 调整小图大小
            # sub_image = sub_image.resize((sub_image_width, sub_image_height))
            # 计算小图在大图中的位置
            x = (ii % num_cols) * sub_image_width
            y = (ii // num_cols) * sub_image_height
            # 将小图粘贴到大图上
            large_image.paste(sub_image, (x, y))

    # 保存拼接后的大图
        large_image.save(out_path+ '/' + str(i) + '.jpg')

if __name__ == '__main__':
    mk = [str(i) for i in range(40, 70, 10)]
    th = [str(i/10) for i in range(5, 10)]
    out_path = r'E:\GangLiu\AI\data\\20230806BigpicNoClean'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for t in th:
        for m in mk:
            for label in [-1, 1]:
                img_path = fr'E:\GangLiu\AI\data\20230730Test\masked_img\{m}x{m}_20_{t}\{label}'
                generate(img_path, out_path)