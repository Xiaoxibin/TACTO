import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TactileVisionDataset(Dataset):
    def __init__(self, Fruit_type=None, Tactile_scale_ratio=1, Visual_scale_ratio=0.25, video_length=8, data_path='./data'):
        self.data_path = data_path
        self.train_data = []
        self.Fruit_type = Fruit_type
        self.Tactile_scale_ratio = Tactile_scale_ratio
        self.Visual_scale_ratio = Visual_scale_ratio
        self.video_length = video_length

        for fruit in self.Fruit_type:
            root = os.path.join(data_path, fruit)
            label_file = "labels.txt"
            with open(os.path.join(root, label_file), 'r') as fp:
                lines = fp.readlines()
                self.train_data.extend([os.path.join(fruit, line.strip()) for line in lines])

        self.train_data.sort()

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        data_line = self.train_data[index]
        parts = data_line.split()
        path_prefix = os.path.join(self.data_path, parts[0])

        # Load images
        rgb_images_pinching = self.load_images(os.path.join(path_prefix, 'pinching_rgb'))
        rgb_images_sliding = self.load_images(os.path.join(path_prefix, 'sliding_rgb'))
        tactile_images_pinching = self.load_images(os.path.join(path_prefix, 'pinching_tactile'))
        tactile_images_sliding = self.load_images(os.path.join(path_prefix, 'sliding_tactile'))

        # Convert to tensors and add batch dimension
        rgb_images_pinching = torch.cat([self.image_to_tensor(img) for img in rgb_images_pinching], dim=0).unsqueeze(0)
        rgb_images_sliding = torch.cat([self.image_to_tensor(img) for img in rgb_images_sliding], dim=0).unsqueeze(0)
        tactile_images_pinching = torch.cat([self.image_to_tensor(img) for img in tactile_images_pinching], dim=0).unsqueeze(0)
        tactile_images_sliding = torch.cat([self.image_to_tensor(img) for img in tactile_images_sliding], dim=0).unsqueeze(0)

        # Extract labels and thresholds
        labels = torch.tensor([int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
        thresholds = torch.tensor([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])

        return rgb_images_pinching, rgb_images_sliding, tactile_images_pinching, tactile_images_sliding, labels, thresholds

    def load_images(self, folder_path):
        image_files = sorted(os.listdir(folder_path))
        return [os.path.join(folder_path, img) for img in image_files]

    def image_to_tensor(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((150, 200)),  # 根据实际情况调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

# 示例使用
def main():
    dataset = TactileVisionDataset(Fruit_type=['apple1'], data_path='/home/xxb/Downloads/form_closed_data')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader):
        rgb_images_pinching = data[0]
        rgb_images_sliding = data[1]
        tactile_images_pinching = data[2]
        tactile_images_sliding = data[3]
        labels = data[4]
        thresholds = data[5]

        print("Batch", i)
        print("rgb_images_pinching shape:", rgb_images_pinching.shape)
        print("rgb_images_sliding shape:", rgb_images_sliding.shape)
        print("tactile_images_pinching shape:", tactile_images_pinching.shape)
        print("tactile_images_sliding shape:", tactile_images_sliding.shape)
        print("labels:", labels)
        print("thresholds:", thresholds)

        # 这里只是一个示例，实际测试时应调用模型进行预测

if __name__ == "__main__":
    main()
