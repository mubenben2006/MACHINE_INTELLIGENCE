import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt


class ImageAugment:
    """图像增强类，提供多种数据增强方法"""

    def __init__(self):
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 训练时的增强变换
        self.train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=15,  # 旋转
                translate=(0.1, 0.1),  # 平移
                scale=(0.9, 1.1),  # 缩放
                shear=5  # 剪切
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 高级增强
        self.advanced_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15),
                                    scale=(0.85, 1.15), shear=10),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ], p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.2),
            transforms.RandomApply([
                transforms.RandomInvert(p=0.1)
            ], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def apply_augmentation(self, image, aug_type='basic'):
        """应用数据增强

        Args:
            image: PIL Image or Tensor
            aug_type: 'basic', 'advanced', or 'none'
        """
        if isinstance(image, torch.Tensor):
            # 如果是tensor，先转换为PIL Image
            image = transforms.ToPILImage()(image)

        if aug_type == 'basic':
            return self.train_transform(image)
        elif aug_type == 'advanced':
            return self.advanced_transform(image)
        else:
            return self.base_transform(image)

    def generate_augmented_dataset(self, images, labels, num_augmentations=5, aug_type='basic'):
        """生成增强后的数据集

        Args:
            images: 原始图像张量
            labels: 原始标签
            num_augmentations: 每张图像生成的增强数量
            aug_type: 增强类型
        """
        augmented_images = []
        augmented_labels = []

        for i, (img, label) in enumerate(zip(images, labels)):
            # 添加原始图像
            augmented_images.append(img)
            augmented_labels.append(label)

            # 添加增强图像
            img_pil = transforms.ToPILImage()(img)
            for j in range(num_augmentations):
                aug_img = self.apply_augmentation(img_pil, aug_type)
                augmented_images.append(aug_img)
                augmented_labels.append(label)

        return torch.stack(augmented_images), torch.tensor(augmented_labels)

    def visualize_augmentations(self, image, num_samples=8):
        """可视化不同的增强效果"""
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()

        # 原始图像
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        # 各种增强
        aug_types = ['basic', 'advanced'] * 4
        for i in range(1, min(8, num_samples)):
            aug_img = self.apply_augmentation(image, aug_types[i - 1])
            axes[i].imshow(aug_img.squeeze(), cmap='gray')
            axes[i].set_title(f'{aug_types[i - 1]} Aug')
            axes[i].axis('off')

        plt.tight_layout()
        return fig


class CustomImageProcessor:
    """自定义图像处理器，用于处理外部输入"""

    @staticmethod
    def process_external_image(image_path, target_size=(28, 28)):
        """处理外部手写数字图像"""
        # 读取图像
        img = Image.open(image_path).convert('L')  # 转换为灰度图

        # 调整大小
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # 反转颜色（如果需要）
        img_array = np.array(img)
        if np.mean(img_array) > 127:  # 如果背景是白色
            img_array = 255 - img_array  # 反转颜色

        # 归一化
        img_array = img_array.astype(np.float32) / 255.0

        # 转换为tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        # 标准化
        transform = transforms.Normalize((0.1307,), (0.3081,))
        img_tensor = transform(img_tensor)

        return img_tensor

    @staticmethod
    def create_digit_image(digit, size=(28, 28)):
        """创建一个数字的简单图像（用于测试）"""
        img = Image.new('L', size, color=255)  # 白色背景
        draw = ImageDraw.Draw(img)

        # 简单绘制数字（这里只是一个示例，实际应该使用更复杂的方法）
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 计算文本位置使其居中
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        draw.text((x, y), str(digit), fill=0, font=font)

        return img