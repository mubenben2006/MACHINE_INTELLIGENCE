import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    残差块组件
    包含两个卷积层，通过残差连接解决梯度消失问题
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        return F.relu(out)

class ConvNet(nn.Module):
    """
    改进的CNN架构用于手写数字识别
    包含残差块、BatchNorm和全局平均池化
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # 初始特征提取层
        self.prep = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 第一层：下采样到14x14，通道数64
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64)
        )
        
        # 第二层：下采样到7x7，通道数128
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128)
        )
        
        # 分类器：全局平均池化 + Dropout + 全连接
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，增强空间鲁棒性
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.classifier(x)
        return x

class ResidualCNN(nn.Module):
    """
    更深层的残差CNN架构（备选方案）
    包含3个残差块，性能更好但训练更慢
    """
    def __init__(self, num_classes=10):
        super(ResidualCNN, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 残差块组
        self.res_block1 = ResidualBlock(32)
        self.res_block2 = ResidualBlock(32)
        
        # 下采样层1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 残差块组2
        self.res_block3 = ResidualBlock(64)
        self.res_block4 = ResidualBlock(64)
        
        # 下采样层2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 残差块组3
        self.res_block5 = ResidualBlock(128)
        self.res_block6 = ResidualBlock(128)
        
        # 全局池化和分类
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 第一组残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 下采样
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 第二组残差块
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # 下采样
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 第三组残差块
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        # 分类
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def get_model(model_type='convnet', num_classes=10):
    """
    获取模型实例
    
    Args:
        model_type: 'convnet' 或 'residual'
        num_classes: 类别数量
    """
    if model_type == 'convnet':
        return ConvNet(num_classes=num_classes)
    elif model_type == 'residual':
        return ResidualCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
