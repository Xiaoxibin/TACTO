import os
import re
import random
import cv2
import torch
from torch.utils import data
from torch import nn

from torch.utils.data.dataloader import DataLoader


# For creating a custom dataset: it needs to contain three funcs: __init__, __len__, __getitem__
# Default: no scale ratio
class Tactile_Vision_dataset(data.Dataset):
    def __init__(self, Fruit_type=None, Tactile_scale_ratio=1, Visual_scale_ratio=0.25, video_length=8,
                 data_path='./data'):
        self.data_path = data_path
        self.label_files = []
        self.train_data = []
        self.Fruit_type = Fruit_type
        self.Tactile_scale_ratio = Tactile_scale_ratio
        self.Visual_scale_ratio = Visual_scale_ratio
        self.video_length = video_length
        for fruit in self.Fruit_type:
            data_path_fruit = data_path + '/' + fruit
            root = data_path + '/' + fruit
            label_file = "labels.txt"
            # 以只读模式打开该水果类型文件夹下的 'labels.txt' 文件以读取标签信息
            fp = open(os.path.join(root, label_file), 'r')
            # 读取文件中的每一行
            lines = fp.readlines()
            self.train_data.extend([fruit + '/' + line.replace('\n', '') for line in lines])
        self.train_data.sort()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):  #need to be defined to let data_loader work
        train_data = self.train_data[index].split()
        # 通过拆分得到 train_data 字符串列表
        #print(train_data[0])  #输出 apple/apple2/Data2
        status = int(train_data[1])
        # print(status)         输出 0
        label = torch.tensor([status]).long()
        label = torch.squeeze(label)

        # 修改这里，将 thresh 解析为 8 个数值的列表
        thresholds = list(map(float, train_data[2:2 + self.video_length]))
        Thresh = torch.tensor(thresholds).float()
        # print(Thresh)
        # print(Thresh.shape)
        # Thresh = torch.squeeze(Thresh)
        output_tactile_imgs_pinching = []
        output_rgb_imgs_pinching = []
        output_tactile_imgs_sliding = []
        output_rgb_imgs_sliding = []
        data_dir = self.data_path + '/' + train_data[0]
        rgb_img_paths_pinching = []
        rgb_img_paths_sliding = []
        for root, dirs, files in os.walk(data_dir + '/Grasping/RealSense', topdown=True):
            for file in files:
                if file.endswith('.png'):  # select camera images
                    rgb_img_paths_pinching.append(os.path.join(root, file))
        for root, dirs, files in os.walk(data_dir + '/Sliding/RealSense', topdown=True):
            for file in files:
                if file.endswith('.png'):  # select camera images
                    rgb_img_paths_sliding.append(os.path.join(root, file))
        rgb_img_paths_pinching.sort()
        rgb_img_paths_sliding.sort()
        rgb_img_paths_selected_pinching = []
        rgb_img_paths_selected_sliding = []
        grasping_start_image = [0, 3, 6, 9, 12, 15, 18, 21]  # customized image index sequence
        sliding_start_image = [8, 11, 14, 17, 20, 23, 26, 29]  #不太懂什么意思
        # select given image for the purpose for comparison
        # try to select discrete images
        index = 0
        while (len(rgb_img_paths_selected_pinching) < self.video_length):  # 8 frames per time (LSTM)
            if index in grasping_start_image:
                rgb_img_paths_selected_pinching.append(rgb_img_paths_pinching[index])
            index += 1
        index = 0
        while (len(rgb_img_paths_selected_sliding) < self.video_length):  # 8 frames per time (LSTM)
            if index in sliding_start_image:
                rgb_img_paths_selected_sliding.append(rgb_img_paths_sliding[index])
            index += 1

        if len(rgb_img_paths_selected_pinching) < self.video_length or len(rgb_img_paths_selected_sliding) < self.video_length:
            raise ValueError(f"Not enough images for video length {self.video_length}")
        
        
        index = 0
        for rgb_img_path in rgb_img_paths_selected_pinching:
            tactile_img_path = rgb_img_path.replace('RealSense',
                                                    'Gelsight')  #替换路径中的 'RealSense' 为 'Gelsight'，以获取相应的触觉图像路径
            # 读取 RGB 图像和触觉图像
            rgb_img = cv2.imread(rgb_img_path)
            tactile_img = cv2.imread(tactile_img_path)
            visual_size = rgb_img.shape  # 480, 640, 3 (width, height, channel)
            tactile_size = tactile_img.shape
            # new width / new height = 480 / 640 * scale_percent

            # commented lines are for the attn visual using pretrained K400 models
            # the only goal is to test the attn method on the released dataset
            rgb_img_resized = cv2.resize(rgb_img, (
            int(visual_size[1] * self.Visual_scale_ratio), int(visual_size[0] * self.Visual_scale_ratio)),
                                         interpolation=cv2.INTER_AREA)
            # rgb_img_resized = cv2.resize(rgb_img,(224, 224),interpolation=cv2.INTER_AREA)
            tactile_img_resized = cv2.resize(tactile_img, (
            int(tactile_size[1] * self.Tactile_scale_ratio), int(tactile_size[0] * self.Tactile_scale_ratio)),
                                             interpolation=cv2.INTER_AREA)
            visual_size = rgb_img_resized.shape
            tactile_size = tactile_img_resized.shape
            # size = tactile_img_resized.shape
            rgb_img_tensor = torch.from_numpy(rgb_img_resized.transpose(2, 0, 1)).float()

            #turn into a tensor (3, 240, 320)  -> resized one
            tactile_img_tensor = torch.from_numpy(tactile_img_resized.transpose(2, 0, 1)).float()
            if index == 0:
                output_rgb_imgs_pinching = rgb_img_tensor[None, :]
                output_tactile_imgs_pinching = tactile_img_tensor[None, :]
            else:
                output_rgb_imgs_pinching = torch.cat([output_rgb_imgs_pinching, rgb_img_tensor[None, :]], dim=0)
                output_tactile_imgs_pinching = torch.cat([output_tactile_imgs_pinching, tactile_img_tensor[None, :]],
                                                         dim=0)
            index += 1
        index = 0
        for rgb_img_path in rgb_img_paths_selected_sliding:
            tactile_img_path = rgb_img_path.replace('RealSense', 'Gelsight')
            rgb_img = cv2.imread(rgb_img_path)
            tactile_img = cv2.imread(tactile_img_path)
            visual_size = rgb_img.shape  # 480, 640, 3 (width, height, channel)
            tactile_size = tactile_img.shape
            # new width / new height = 480 / 640 * scale_percent

            # commented lines are for the attn visual using pretrained K400 models
            # the only goal is to test the attn method on the released dataset
            rgb_img_resized = cv2.resize(rgb_img, (
            int(visual_size[1] * self.Visual_scale_ratio), int(visual_size[0] * self.Visual_scale_ratio)),
                                         interpolation=cv2.INTER_AREA)
            # rgb_img_resized = cv2.resize(rgb_img,(224, 224),interpolation=cv2.INTER_AREA)
            tactile_img_resized = cv2.resize(tactile_img, (
            int(tactile_size[1] * self.Tactile_scale_ratio), int(tactile_size[0] * self.Tactile_scale_ratio)),
                                             interpolation=cv2.INTER_AREA)
            visual_size = rgb_img_resized.shape
            tactile_size = tactile_img_resized.shape
            # size = tactile_img_resized.shape

            # 调整大小后的图像转换成张量，并进行维度转换
            rgb_img_tensor = torch.from_numpy(rgb_img_resized.transpose(2, 0, 1)).float()

            #turn into a tensor (3, 240, 320)  -> resized one
            tactile_img_tensor = torch.from_numpy(tactile_img_resized.transpose(2, 0, 1)).float()
            if index == 0:
                output_rgb_imgs_sliding = rgb_img_tensor[None, :]
                output_tactile_imgs_sliding = tactile_img_tensor[None, :]
            else:
                output_rgb_imgs_sliding = torch.cat([output_rgb_imgs_sliding, rgb_img_tensor[None, :]], dim=0)
                # [None,:] -> 在张量的第一个维度(channel 维度)增加一个新的维度；dim=0 -> 在第一个维度上拼接张量(8)
                output_tactile_imgs_sliding = torch.cat([output_tactile_imgs_sliding, tactile_img_tensor[None, :]],
                                                        dim=0)
            index += 1
        # print(output_rgb_imgs_pinching.transpose(0, 1).shape)  # [8, 3, 120, 160]
        return output_rgb_imgs_pinching.transpose(0, 1), output_rgb_imgs_sliding.transpose(0,1), output_tactile_imgs_pinching.transpose(0, 1), output_tactile_imgs_sliding.transpose(0, 1), label, Thresh  # rgb images; visual images; label


if __name__ == "__main__":
    # set a global dataset path
    train_dataset = Tactile_Vision_dataset(Fruit_type=['apple', ], data_path='/home/xxb/Downloads/thresh_data')
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                   num_workers=4)

    data = next(iter(train_data_loader))
    print("Print feed-forward test results:")
    print(data[0].shape)  # visual pinching
    print(data[1].shape)  # visual sliding
    print(data[2].shape)  # tactile pinching
    print(data[3].shape)  # tactile sliding
    print(data[4].shape)  # label
    print(data[5].shape)  # threshold

    # 使用len函数获取信息并输出
    length = len(train_dataset)
    print("Length of train data:", length)

    # data = train_dataset[0][0]
    # print(data.shape)
