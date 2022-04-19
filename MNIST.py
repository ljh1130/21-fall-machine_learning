import struct # 字节字符串和python原生数据类型之间转换
import os
import numpy as np

# 全连接层
class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
    def init_param(self, std=0.01): # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale = std ,
                                       size = (self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input): # 前向传播的计算
        self.input = input
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output
    def backward(self, top_diff): # 反向传播的计算
        # print(top_diff.size)
        self.d_weight = self.input.T.dot(top_diff)
        # print(self.d_weight.size)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff
    def update_param(self, lr): # 参数更新
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias
    def load_param(self, weight, bias): # 参数加载
        self. weight = weight
        self. bias = bias
    def save_param(self): # 参数保存
        return self.weight, self.bias

# ReLU层
class ReLULayer(object):
    def forward(self, input): #前向传播计算
        self.input = np.maximum(0, input)
        output = self.input
        return output
    def backward (self, top_diff): # 反向传播的计算
        top_diff[self.input <= 0] = 0
        bottom_diff = top_diff
        return bottom_diff

# Softmax损失层
class SoftmaxLossLayer(object):
    def forward(self, input): # 前向传播的计算
        print(input)
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp)
        return self.prob
    def get_loss (self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onchot = np.zeros_like(self.prob)
        self.label_onchot[np.arange(self.batch_size), label] = 1.0
        # print(self.prob)
        loss = -np.sum(np.log(self.prob) * self.label_onchot) / self.batch_size
        return loss
    def backward(self): # 反向传播的计算
        bottom_diff = (self.prob - self.label_onchot) / self.batch_size
        return bottom_diff

# 三层神经网络结构
class MNIST_MLP(object):
    def __init__(self, batch_size = 32, input_size = 784 , hidden1 = 32 , hidden2 = 16 ,
        out_classes = 10 , lr = 0.01 , max_epoch = 2 , print_iter = 32):
        # 神经网络初始化
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    # 数据加载模块
    def load_mnist(self, file_dir, is_images):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_images == 1:  # 读取图像数据
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_sizc = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_sizc) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data

    def load_data(self):
        train_images = self.load_mnist(os.path.join('/Users/lijihang/PycharmProjects/机器学习实验/mnist_test/mnist_data', 'train-images-idx3-ubyte'),1)
        train_labels = self.load_mnist(os.path.join('/Users/lijihang/PycharmProjects/机器学习实验/mnist_test/mnist_data', 'train-labels-idx1-ubyte'), 0)
        test_images = self.load_mnist(os.path.join('/Users/lijihang/PycharmProjects/机器学习实验/mnist_test/mnist_data', 't10k-images-idx3-ubyte'), 1)
        test_labels = self.load_mnist(os.path.join('/Users/lijihang/PycharmProjects/机器学习实验/mnist_test/mnist_data', 't10k-labels-idx1-ubyte'), 0)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def build_model(self): # 建立网络结构
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]
    def init_model (self): # 神经网络参数初始化
        for layer in self.update_layer_list:
            layer.init_param ()

    # 训练模块
    def forward(self, input): # 神经网络的前向传播
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        prob = self.softmax.forward(h3)
        return prob

    def backward(self): # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh3 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr): # 神经网络参数更新
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir): # 保存神经网络参数
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    def train(self): # 训练函数主体
        max_batch = self.train_data.shape[0] // self.batch_size
        for idx_epoch in range(self.max_epoch):
            np.random.shuffle(self.train_data)
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size: (idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size: (idx_batch + 1) * self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))
    # 网络推断模块
    def load_model(self, pararm_dir): # 加载神经网络参数
        params = np.load(pararm_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])

    def evaluate(self): # 推断函数主体
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(self.test_data.shape[0]//self.batch_size):
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:,-1])
        print('Accuracy in test set：%.2f%%' % (accuracy * 100))

# 实验流程
if __name__ == '__main__':
    h1, h2, e = 32, 16, 1
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.train()
    mlp.save_model('mlp-%d-%d-%depoch.npy' % (h1, h2 ,e))
    mlp.load_model('mlp-%d-%d-%depoch.npy' % (h1, h2 ,e))
    mlp.evaluate()