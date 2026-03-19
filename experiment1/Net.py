# Net.py
import numpy as np
from ImageAugmenter import ImageAugmenter
from MathTools import softmax, ReLU, ReLU_derivative, cross_entropy

class Net:
    def __init__(self, input_size, output_size, linears=None):
        """
        初始化神经网络
        
        参数:
        input_size: 输入层维度
        output_size: 输出层维度
        linears: 隐藏层维度列表
        """
        dims = [input_size] + (linears or []) + [output_size]
        self.layers = []
        self.cache = []
        
        for i in range(len(dims) - 1):
            std = np.sqrt(2.0 / dims[i])  # He初始化
            self.layers.append({
                'W': np.random.randn(dims[i], dims[i + 1]) * std,
                'b': np.zeros((1, dims[i + 1])),
                'mW': np.zeros((dims[i], dims[i + 1])),  # 一阶矩
                'vW': np.zeros((dims[i], dims[i + 1])),  # 二阶矩
                'mb': np.zeros((1, dims[i + 1])),
                'vb': np.zeros((1, dims[i + 1]))
            })
        self.t = 0  # Adam迭代计数器

    def forward(self, X, training=True, dropout_rate=0.1):
        """
        前向传播
        
        参数:
        X: 输入数据
        training: 是否训练模式（影响dropout）
        dropout_rate: dropout概率
        
        返回:
        预测结果
        """
        A = X
        self.cache = []
        self.cache.append({'A': X, 'Z': None})
        
        for i in range(len(self.layers)):
            layer = self.layers[i]
            Z = A @ layer['W'] + layer['b']
            
            if i == len(self.layers) - 1:  # 输出层
                A = softmax(Z)
            else:  # 隐藏层
                A = ReLU(Z)
                if training:
                    # Dropout
                    mask = (np.random.rand(*A.shape) > dropout_rate) / (1 - dropout_rate)
                    A *= mask
                    self.cache[-1]['mask'] = mask
                    
            self.cache.append({'A': A, 'Z': Z})
        
        return A

    def backward(self, y_true, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        反向传播（Adam优化器）
        
        参数:
        y_true: 真实标签
        lr: 学习率
        beta1, beta2: Adam超参数
        epsilon: 防止除零
        """
        m = y_true.shape[0]
        self.t += 1
        dz = self.cache[-1]['A'] - y_true
        
        for i in reversed(range(len(self.layers))):
            A_prev = self.cache[i]['A']
            layer = self.layers[i]
            
            dW = (A_prev.T @ dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            # Adam更新
            layer['mW'] = beta1 * layer['mW'] + (1 - beta1) * dW
            layer['mb'] = beta1 * layer['mb'] + (1 - beta1) * db
            layer['vW'] = beta2 * layer['vW'] + (1 - beta2) * (dW ** 2)
            layer['vb'] = beta2 * layer['vb'] + (1 - beta2) * (db ** 2)

            # 偏置修正
            mW_hat = layer['mW'] / (1 - beta1 ** self.t)
            mb_hat = layer['mb'] / (1 - beta1 ** self.t)
            vW_hat = layer['vW'] / (1 - beta2 ** self.t)
            vb_hat = layer['vb'] / (1 - beta2 ** self.t)

            # 更新权重
            layer['W'] -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
            layer['b'] -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
            
            if i > 0:  # 不是输入层
                dz = (dz @ layer['W'].T) * ReLU_derivative(self.cache[i]['Z'])
                if 'mask' in self.cache[i - 1]:
                    dz *= self.cache[i - 1]['mask']

    def train(self, x_train, y_train, batch_size=32, epochs=100, lr=0.001, 
              counts=100, augment_prob=0.5):
        """
        训练模型
        
        参数:
        x_train: 训练数据
        y_train: 训练标签
        batch_size: 批次大小
        epochs: 训练轮数
        lr: 初始学习率
        counts: 打印间隔
        augment_prob: 数据增强概率
        
        返回:
        训练历史记录
        """
        n_samples = len(x_train)
        current_lr = lr
        augmenter = ImageAugmenter(image_shape=(28, 28), random_seed=42)
        
        history = {'loss': [], 'epoch_loss': []}
        
        for epoch in range(epochs):
            # 学习率衰减
            if epoch > 0 and epoch % 10 == 0:
                current_lr *= 0.5
                print(f"--- 学习率衰减为: {current_lr:.5f} ---")
            
            # 打乱数据
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            
            running_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                xb = x_train[batch_indices]
                yb = y_train[batch_indices]
                
                # 数据增强
                if np.random.rand() < augment_prob:
                    xb = augmenter.augment_batch(xb)
                
                # 前向传播
                y_pred = self.forward(xb)
                
                # 反向传播
                self.backward(yb, current_lr)
                
                # 计算损失
                current_loss = cross_entropy(yb, y_pred)
                running_loss += current_loss
                n_batches += 1
                
                # 打印进度
                if i % counts == 0:
                    avg_loss = running_loss / n_batches
                    print(f'Epoch:{epoch} | Batch:{i // batch_size} | '
                          f'Avg Loss:{avg_loss:.4f} | LR:{current_lr:.5f}')
                    history['loss'].append(avg_loss)
            
            # 记录epoch损失
            epoch_loss = running_loss / n_batches
            history['epoch_loss'].append(epoch_loss)
            print(f'Epoch {epoch} 完成，平均损失: {epoch_loss:.4f}')
        
        return history

    def evaluate(self, x_test, y_test):
        """
        评估模型准确率
        
        参数:
        x_test: 测试数据
        y_test: 测试标签（one-hot编码）
        
        返回:
        准确率
        """
        y_pred = self.forward(x_test, training=False)
        predictions = np.argmax(y_pred, axis=-1)
        labels = np.argmax(y_test, axis=-1)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def predict(self, x):
        """
        预测
        
        参数:
        x: 输入数据
        
        返回:
        预测概率
        """
        return self.forward(x, training=False)

    def save_model(self, filepath):
        """保存模型参数"""
        model_data = {
            'layers': self.layers,
            't': self.t
        }
        np.save(filepath, model_data)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """加载模型参数"""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.layers = model_data['layers']
        self.t = model_data['t']
        print(f"模型已从 {filepath} 加载")
