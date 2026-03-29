import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
from tqdm import tqdm
import os
from PIL import Image

from net import ImprovedCNN, SimpleCNN
from image_augment import ImageAugment


class HandwrittenDigitsRecognizer:
    def __init__(self, model_type='improved', device=None):
        """
        初始化识别器
        Args:
            model_type: 'improved' 或 'simple'
            device: 设备类型
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 选择模型架构
        if model_type == 'improved':
            self.model = ImprovedCNN(num_classes=10).to(self.device)
        else:
            self.model = SimpleCNN(num_classes=10).to(self.device)

        self.image_augment = ImageAugment()

        # 训练历史记录
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def load_mnist_data(self, batch_size=64, use_augmentation=True,
                        augmentation_type='advanced', augmented_samples=0):
        """加载MNIST数据集，可选择是否使用增强"""

        # 基础变换
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载原始数据
        print("加载原始MNIST数据...")
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                       transform=base_transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True,
                                      transform=base_transform)

        if use_augmentation and augmented_samples > 0:
            print(f"正在创建增强数据集，每张图像生成 {augmented_samples} 个增强样本...")

            # 收集所有训练数据
            all_images = []
            all_labels = []

            # 添加原始数据
            for i in tqdm(range(len(train_dataset)), desc="处理原始数据"):
                img, label = train_dataset[i]
                all_images.append(img)
                all_labels.append(label)

            # 创建增强数据
            print("生成增强数据...")
            for i in tqdm(range(min(5000, len(train_dataset))), desc="生成增强样本"):  # 限制增强数量
                img, label = train_dataset[i]
                # 将tensor转换为PIL图像
                img_pil = transforms.ToPILImage()(img)

                # 生成增强样本
                for _ in range(augmented_samples):
                    if augmentation_type == 'basic':
                        aug_img = self.image_augment.apply_augmentation(img_pil, 'basic')
                    else:
                        aug_img = self.image_augment.apply_augmentation(img_pil, 'advanced')

                    all_images.append(aug_img)
                    all_labels.append(label)

            # 创建新的训练数据集
            print(f"创建TensorDataset，总样本数: {len(all_images)}")
            train_dataset = TensorDataset(
                torch.stack(all_images),
                torch.tensor(all_labels, dtype=torch.long)
            )

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0)

        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")

        return train_loader, test_loader

    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="训练")):
            # 确保数据类型正确
            data = data.float().to(self.device)
            target = target.long().to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="验证"):
                data = data.float().to(self.device)
                target = target.long().to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = running_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=20, lr=0.001):
        """训练模型"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=3)

        print(f"\n开始训练，共 {epochs} 个epochs...")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # 调整学习率
            scheduler.step(val_loss)

            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")

    def evaluate_model(self, test_loader):
        """全面评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="评估"):
                data = data.float().to(self.device)
                target = target.long().to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # 计算各种指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 准确率
        accuracy = np.mean(all_predictions == all_targets) * 100

        # 精确率、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions,
                                                                   average='weighted')

        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)

        # 分类报告
        class_report = classification_report(all_targets, all_predictions)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report
        }

        return metrics

    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率')
        ax2.plot(self.val_accuracies, label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, cm, save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()

    def predict_external_image(self, image_path=None):
        """预测外部手写数字图像"""
        self.model.eval()

        if not image_path:
            raise ValueError("请提供图像路径")

        # 读取并处理图像
        img = Image.open(image_path).convert('L')  # 转换为灰度图

        # 调整大小为28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 反转颜色（如果需要）
        img_array = np.array(img)
        if np.mean(img_array) > 127:  # 如果背景是白色
            img_array = 255 - img_array  # 反转颜色

        # 转换为tensor并归一化
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度

        # 标准化
        transform = transforms.Normalize((0.1307,), (0.3081,))
        img_tensor = transform(img_tensor)

        # 预测
        with torch.no_grad():
            img_tensor = img_tensor.float().to(self.device)
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence, probabilities.cpu().numpy()

    def show_augmentation_examples(self):
        """显示增强图像示例"""
        # 从测试集中取一些图像
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=base_transform)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        for i in range(5):
            # 原始图像
            img, label = test_dataset[i]
            # 反归一化显示
            img_display = img.squeeze().numpy() * 0.3081 + 0.1307
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f'原始数字: {label}')
            axes[0, i].axis('off')

            # 增强后的图像
            img_pil = transforms.ToPILImage()(img)
            aug_img = self.image_augment.apply_augmentation(img_pil, 'advanced')
            aug_img_display = aug_img.squeeze().numpy() * 0.3081 + 0.1307
            axes[1, i].imshow(aug_img_display, cmap='gray')
            axes[1, i].set_title('增强后')
            axes[1, i].axis('off')

        plt.tight_layout()
        return fig

    def save_model(self, path='model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, path)
        print(f"模型已保存到 {path}")

    def load_model(self, path='model.pth'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        print(f"模型已从 {path} 加载")


def main():
    """主函数"""
    print("=" * 50)
    print("手写数字识别系统")
    print("=" * 50)

    # 创建识别器
    recognizer = HandwrittenDigitsRecognizer(model_type='improved')

    # 选择模式
    print("\n请选择操作模式:")
    print("1. 训练新模型")
    print("2. 加载已有模型并进行预测")

    mode = input("请输入选择 (1/2): ").strip()

    if mode == '1':
        # 训练模式
        print("\n训练配置:")
        use_augmentation = input("是否使用数据增强? (y/n): ").strip().lower() == 'y'

        if use_augmentation:
            aug_samples = int(input("每张图像生成几个增强样本? (建议: 2-3): ") or "2")
            aug_type = input("增强类型 (basic/advanced): ").strip() or "basic"
        else:
            aug_samples = 0
            aug_type = 'basic'

        batch_size = int(input("批次大小? (建议: 64): ") or "64")
        epochs = int(input("训练轮数? (建议: 5-10): ") or "5")

        # 加载数据
        print("\n加载数据...")
        train_loader, test_loader = recognizer.load_mnist_data(
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            augmentation_type=aug_type,
            augmented_samples=aug_samples
        )

        # 训练模型
        recognizer.train(train_loader, test_loader, epochs=epochs, lr=0.001)

        # 评估模型
        print("\n评估模型...")
        metrics = recognizer.evaluate_model(test_loader)

        print(f"\n最终评估结果:")
        print(f"准确率: {metrics['accuracy']:.2f}%")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")
        print(f"\n分类报告:\n{metrics['classification_report']}")

        # 可视化
        recognizer.plot_training_history()
        plt.show()

        recognizer.plot_confusion_matrix(metrics['confusion_matrix'], 'confusion_matrix.png')

        # 显示增强示例
        if use_augmentation:
            print("\n显示增强图像示例...")
            recognizer.show_augmentation_examples()
            plt.show()

        # 保存模型
        save_choice = input("\n是否保存模型? (y/n): ").strip().lower()
        if save_choice == 'y':
            recognizer.save_model('mnist_model.pth')

    elif mode == '2':
        # 预测模式
        model_path = input("请输入模型路径 (默认: mnist_model.pth): ").strip() or "mnist_model.pth"

        if os.path.exists(model_path):
            recognizer.load_model(model_path)
            print("模型加载成功!")
        else:
            print(f"模型文件 {model_path} 不存在，请先训练模型!")
            return

        # 预测循环
        while True:
            print("\n" + "=" * 30)
            print("预测选项:")
            print("1. 从文件预测")
            print("2. 退出")

            choice = input("请选择: ").strip()

            if choice == '1':
                img_path = input("请输入图像路径: ").strip()
                if os.path.exists(img_path):
                    try:
                        pred, conf, probs = recognizer.predict_external_image(image_path=img_path)
                        print(f"\n预测结果: {pred}")
                        print(f"置信度: {conf:.4f}")

                        # 显示概率分布
                        plt.figure(figsize=(10, 4))
                        plt.bar(range(10), probs[0])
                        plt.xlabel('数字')
                        plt.ylabel('概率')
                        plt.title('预测概率分布')
                        plt.show()
                    except Exception as e:
                        print(f"预测出错: {e}")
                else:
                    print("文件不存在!")

            elif choice == '2':
                print("再见!")
                break

            else:
                print("无效选择!")

    else:
        print("无效选择!")


if __name__ == "__main__":
    main()