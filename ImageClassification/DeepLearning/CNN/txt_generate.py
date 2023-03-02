import os

def generate(img_path):
    f = open("label.txt", 'a')
    label_list = os.listdir(img_path)
    print(label_list)
    count = -1

    for label in label_list:
        count += 1
        root = os.path.join(img_path, label)
        img_root = os.listdir(root)
        for file_root in img_root:
            if file_root.endswith('jpg'):
                text = os.path.join(root, file_root) + ' ' + str(count) + '\n'
                f.write(text)
    f.close()

if __name__ == '__main__':
    generate("../../../data/mydata/test")