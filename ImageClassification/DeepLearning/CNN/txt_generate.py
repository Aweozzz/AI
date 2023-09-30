import os
import sys

def generate(img_path):
    f = open("label.txt", 'w')
    label_list = os.listdir(img_path)
    print(label_list)

    for label in label_list:

        root = os.path.join(img_path, label)
        img_root = os.listdir(root)
        for file_root in img_root:
            if file_root.endswith('jpg') or file_root.endswith('png') or file_root.endswith('tif'):
                text = os.path.join(root, file_root) + ' ' + label + '\n'
                f.write(text)
    f.close()

if __name__ == '__main__':
    args = sys.argv[1:]
    generate(args[0])