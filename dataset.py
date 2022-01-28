import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MidlineDataset(Dataset):
    def __init__(self, img_dir, txt_file):
        super(MidlineDataset, self).__init__()
        self.img_dir = img_dir
        self.landmarks_frame = pd.read_csv(txt_file, header=None)  # txt文件
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, item):
        # image
        img_path = os.path.join(
            self.img_dir,
            self.landmarks_frame.iloc[item, 0]
        )
        image = Image.open(img_path)  # .convert('L')
        W, H = image.size  # W:94 H:60

        # label
        landmarks = self.landmarks_frame.iloc[item, 1:]
        landmarks = np.array(landmarks, dtype='float32')
        # 坐标归一化
        landmarks[:5] = landmarks[:5] / W
        landmarks[5:] = landmarks[5:] / H

        return self.transform(image), torch.tensor(landmarks)


def main():
    img_dir = "data/train_images"
    txt_file = "data/train.txt"
    dataset = MidlineDataset(img_dir, txt_file)
    print(len(dataset))
    _image = dataset[0][0].numpy()
    _label = dataset[0][1].numpy()
    print(_image.shape, _label.shape)  # H:60 W:94
    print(_image.max(), _image.min())
    print(_label)


if __name__ == "__main__":
    main()




