import numpy as np
import matplotlib.pyplot as plt
from Model.four_bit_cell import Realistic_RRAM_Processor
# 复用之前的 Realistic_RRAM_Processor 类逻辑 (简略表示)
class RRAM_App_Simulator(Realistic_RRAM_Processor):
    def classify(self, input_vector, weight_matrix):
        # 硬件推理：得到 4 个神经元的输出电平
        raw_outputs = self.run(input_vector, weight_matrix)
        # Softmax 思想：输出最大的那个索引即为识别结果
        return np.argmax(raw_outputs), raw_outputs

# 1. 定义 4 种模式的理想权重 (归一化到 -1 ~ 1)
# 每一列代表一个模式探测器
# 模式：水平线, 垂直线, 对角线, 全亮
weights = np.array([
    [ 1,  1,  1,  1], # 模式1: [1,1,0,0] -> 水平
    [ 1, -1, -1,  1], # 模式2: [1,0,1,0] -> 垂直
    [-1,  1,  1, -1], # 模式3: [1,0,0,1] -> 对角
    [-1, -1,  1,  1]  # 模式4: [1,1,1,1] -> 全亮
]).T

# 2. 初始化带噪声的处理器
# 设定 10% 的电导波动，模拟一个“状态一般”的 RRAM 芯片
hw_processor = RRAM_App_Simulator(g_sigma=0.10, v_noise_std=0.02)

# 3. 生成测试数据 (带噪声的输入图像)
def get_test_data(n_samples=50):
    test_inputs = []
    test_labels = []
    patterns = [
        [1, 1, 0, 0], # Horizontal
        [1, 0, 1, 0], # Vertical
        [1, 0, 0, 1], # Diagonal
        [1, 1, 1, 1]  # All
    ]
    for _ in range(n_samples):
        p_idx = np.random.randint(0, 4)
        # 加入一点输入端的随机噪声
        noisy_input = np.clip(np.array(patterns[p_idx]) + np.random.normal(0, 0.1, 4), 0, 1)
        test_inputs.append(noisy_input)
        test_labels.append(p_idx)
    return test_inputs, test_labels

# 4. 执行测试
inputs, labels = get_test_data(200)
correct = 0
results_map = []

for i in range(len(inputs)):
    pred, scores = hw_processor.classify(inputs[i], weights)
    results_map.append(scores)
    if pred == labels[i]:
        correct += 1

accuracy = correct / len(inputs)
print(f"RRAM 芯片分类准确率 (含10%波动): {accuracy*100:.2f}%")

# 5. 可视化：输出得分矩阵 (Confusion Visualization)
plt.figure(figsize=(10, 4))
plt.title(f"RRAM Hardware Output Scores (Accuracy: {accuracy*100:.1f}%)")
plt.imshow(np.array(results_map)[:40].T, aspect='auto', cmap='viridis')
plt.colorbar(label='Neuron Output Amplitude')
plt.xlabel("Test Sample Index")
plt.ylabel("Pattern Neuron (0-3)")
plt.show()