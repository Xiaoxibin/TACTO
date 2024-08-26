import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class GraspingSlidingDataset(Dataset):
    def __init__(self, data_path, transform_rgb=None, transform_tactile=None):
        self.data_path = data_path
        self.transform_rgb = transform_rgb
        self.transform_tactile = transform_tactile

        self.rgb_pinching_paths = []
        self.rgb_sliding_paths = []
        self.tactile_pinching_paths = []
        self.tactile_sliding_paths = []

        self.load_paths('grasping', 'rgb', self.rgb_pinching_paths)
        self.load_paths('grasping', 'tactile', self.tactile_pinching_paths)
        self.load_paths('sliding', 'rgb', self.rgb_sliding_paths)
        self.load_paths('sliding', 'tactile', self.tactile_sliding_paths)

    def load_paths(self, folder, img_type, path_list):
        folder_path = os.path.join(self.data_path, folder, img_type)
        for img_file in sorted(os.listdir(folder_path)):
            if img_file.endswith('.png') or img_file.endswith('.jpg'):
                path_list.append(os.path.join(folder_path, img_file))

    def __len__(self):
        return len(self.rgb_pinching_paths) // 8

    def __getitem__(self, idx):
        start_idx = idx * 8
        end_idx = start_idx + 8

        rgb_imgs_pinching = [self.load_image(self.rgb_pinching_paths[i], self.transform_rgb) for i in range(start_idx, end_idx)]
        rgb_imgs_sliding = [self.load_image(self.rgb_sliding_paths[i], self.transform_rgb) for i in range(start_idx, end_idx)]
        tactile_imgs_pinching = [self.load_image(self.tactile_pinching_paths[i], self.transform_tactile) for i in range(start_idx, end_idx)]
        tactile_imgs_sliding = [self.load_image(self.tactile_sliding_paths[i], self.transform_tactile) for i in range(start_idx, end_idx)]

        output_rgb_imgs_pinching = torch.cat(rgb_imgs_pinching, dim=0).transpose(0, 1)
        output_rgb_imgs_sliding = torch.cat(rgb_imgs_sliding, dim=0).transpose(0, 1)
        output_tactile_imgs_pinching = torch.cat(tactile_imgs_pinching, dim=0).transpose(0, 1)
        output_tactile_imgs_sliding = torch.cat(tactile_imgs_sliding, dim=0).transpose(0, 1)

        return output_rgb_imgs_pinching, output_rgb_imgs_sliding, output_tactile_imgs_pinching, output_tactile_imgs_sliding

    def load_image(self, img_path, transform):
        image = Image.open(img_path).convert('RGB')
        if transform:
            image = transform(image)
        return image.unsqueeze(0)

if __name__ == "__main__":
# 定义图像转换
    transform_rgb = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor()
    ])

    transform_tactile = transforms.Compose([
        transforms.Resize((150, 200)),
        transforms.ToTensor()
    ])

    # 创建数据集实例
    data_path = '/home/xxb/Downloads/data'
    dataset = GraspingSlidingDataset(data_path, transform_rgb=transform_rgb, transform_tactile=transform_tactile)

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 测试DataLoader
    for data in dataloader:
        output_rgb_imgs_pinching, output_rgb_imgs_sliding, output_tactile_imgs_pinching, output_tactile_imgs_sliding = data
        print(output_rgb_imgs_pinching.size())
        print(output_rgb_imgs_sliding.size())
        print(output_tactile_imgs_pinching.size())
        print(output_tactile_imgs_sliding.size())
        break


