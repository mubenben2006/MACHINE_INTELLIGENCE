import numpy as np
def augment_batch(images, max_shift=2):
    """
    对一个 Batch 的图片进行随机平移
    images shape: (batch_size, 784)
    """
    batch_size = images.shape[0]
    # 把 (784,) 还原回 (28, 28)
    imgs = images.reshape(-1, 28, 28)
    augmented_imgs = np.zeros_like(imgs)
    
    for i in range(batch_size):
        # 随机产生平移量 (dy, dx)
        dy, dx = np.random.randint(-max_shift, max_shift + 1, size=2)
        
        # 使用 np.roll 进行平移（循环滚动）
        shifted = np.roll(imgs[i], shift=(dy, dx), axis=(0, 1))
        
        # 将滚回来的边缘清零（防止循环像素干扰）
        if dy > 0: shifted[:dy, :] = 0
        elif dy < 0: shifted[dy:, :] = 0
        if dx > 0: shifted[:, :dx] = 0
        elif dx < 0: shifted[:, dx:] = 0
        
        augmented_imgs[i] = shifted
        
    return augmented_imgs.reshape(batch_size, 784)
class Net:
    def __init__(self,input_size,output_size,linears=None):
        dims=[input_size]+(linears or [])+[output_size]
        self.layers=[]
        self.cache = []
        for i in range(len(dims)-1):
            std = np.sqrt(2.0 / dims[i])#He初始化
            #引入Adam
            self.layers.append({
            'W': np.random.randn(dims[i], dims[i+1]) * std,
            'b': np.zeros((1, dims[i+1])),
            'mW': np.zeros((dims[i], dims[i+1])), # 一阶矩
            'vW': np.zeros((dims[i], dims[i+1])), # 二阶矩
            'mb': np.zeros((1, dims[i+1])),
            'vb': np.zeros((1, dims[i+1]))
            })
        self.t = 0 # 迭代次数计数器，用于偏置修正
    def forward(self,X,training=True, dropout_rate=0.1):
        A=X
        self.cache = []
        self.cache.append({'A':X,'Z':None})
        for i in range(len(self.layers)):
            layer = self.layers[i]
            Z = A @ layer['W'] + layer['b']
            if i==len(self.layers)-1:
                A=self.softmax(Z)
            else:
                A=self.ReLU(Z)
                if training:
                    # 生成 0/1 掩码，保留概率为 (1 - dropout_rate)
                    mask = (np.random.rand(*A.shape) > dropout_rate) / (1 - dropout_rate)
                    A *= mask
                    self.cache[-1]['mask'] = mask # 存起来给 backward 用
            self.cache.append({'A': A, 'Z': Z})
        return A 
    def backward(self,y_true,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        m=y_true.shape[0]
        self.t+=1
        dz = self.cache[-1]['A'] - y_true
        for i in reversed(range(len(self.layers))):
            A_prev=self.cache[i]['A']
            layer=self.layers[i]
            dW=(A_prev.T@dz)/m
            db=np.sum(dz,axis=0,keepdims=True)/m
            # --- Adam 核心逻辑 ---
            # 1. 更新一阶矩 (Momentum)
            layer['mW'] = beta1 * layer['mW'] + (1 - beta1) * dW
            layer['mb'] = beta1 * layer['mb'] + (1 - beta1) * db
            
            # 2. 更新二阶矩 (RMSProp)
            layer['vW'] = beta2 * layer['vW'] + (1 - beta2) * (dW**2)
            layer['vb'] = beta2 * layer['vb'] + (1 - beta2) * (db**2)
            
            # 3. 偏置修正 (Bias Correction) - 解决训练初期 m 和 v 接近 0 的问题
            mW_hat = layer['mW'] / (1 - beta1**self.t)
            mb_hat = layer['mb'] / (1 - beta1**self.t)
            vW_hat = layer['vW'] / (1 - beta2**self.t)
            vb_hat = layer['vb'] / (1 - beta2**self.t)
            
            # 4. 最终更新权重
            layer['W'] -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
            layer['b'] -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
            if i > 0:
                dz = (dz @ layer['W'].T) * self.ReLU_derivative(self.cache[i]['Z'])
                if 'mask' in self.cache[i-1]:
                    dz *= self.cache[i-1]['mask']

    def softmax(self,tensor):
        """Softmax函数"""
        shifted_tensor = tensor - np.max(tensor, axis=-1, keepdims=True) #减去最大值防止exp溢出
        exps = np.exp(shifted_tensor)
        return exps / np.sum(exps, axis=-1, keepdims=True)
    def Z_score(self,tensor, ddof=0):
        """Z-score 标准化函数,按列进行标准化"""
        tensor = np.asarray(tensor, dtype=np.float32)
        mu = tensor.mean(axis=0, keepdims=True)
        variance = np.var(tensor, axis=0, keepdims=True, ddof=ddof)
        sigma = np.sqrt(variance + 1e-8)
        return (tensor - mu) / sigma
    def ReLU(self,tensor):
        return np.maximum(0,tensor)
    def ReLU_derivative(self, Z):
        # ReLU 在 Z > 0 时导数为 1，否则为 0
        return (Z > 0).astype(float)
    def cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    def train(self,x_train,y_train,batch_size,epochs,lr,counts=100):
        n_samples = len(x_train)
        current_lr = lr
        for epoch in range(epochs):
            if epoch > 0 and epoch % 10 == 0:
                current_lr *= 0.5
                print(f"--- 学习率衰减为: {current_lr:.5f} ---")
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            running_loss = 0.0  # 累计损失
            n_batches = 0
            for i in range(0,n_samples,batch_size):
                batch_indices=indices[i:i+batch_size]
                xb=x_train[batch_indices]
                yb=y_train[batch_indices]
                if np.random.rand() > 0.5:
                    xb = augment_batch(xb, max_shift=2)
                y_pred=self.forward(xb)
                self.backward(yb,current_lr)
                current_loss=self.cross_entropy(yb, y_pred)
                running_loss+=current_loss
                n_batches+=1
                if i%counts==0:
                    avg_loss = running_loss / n_batches
                    print(f'Epoch:{epoch} | Batch:{i//batch_size} | Avg Loss:{avg_loss:.4f}')
    def evaluate(self, x_test, y_test):
        y_pred = self.forward(x_test,training=False)
        # 找到概率最大的类别索引
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_test, axis=1)
        # 计算准确率
        accuracy = np.mean(predictions == labels)
        return accuracy