import numpy as np
import matplotlib.pyplot as plt
from Model.non_ideal_model import RRAM_VMM_Processor_4b_nonideal

class Test_RRAM_VMM_Processor:
    def batch_test(self, num_samples=100):
        """
        随机生成测试集并分析误差
        """
        print(f"\n{'=' * 20} 开始批量测试 (N={num_samples}) {'=' * 20}")

        all_ideal = []
        all_actual = []
        errors = []

        for i in range(num_samples):
            # 1. 随机生成输入 (0~1 浮点数)
            vec = np.random.randint(0, 256, 4)
            # 2. 随机生成权重 (0 或 1 整数)
            mat = np.random.randint(0, 16, size=(4,4))

            # 3. 计算理想值 (数学真值)
            ideal_res = np.dot(vec, mat)

            # 4. 计算模拟器值

            processor_instance = RRAM_VMM_Processor_4b_nonideal()
            sim_res = processor_instance.run(vec, mat)

            # 5. 记录数据
            all_ideal.extend(ideal_res)
            all_actual.extend(sim_res)

        # --- 统计结果 ---
        all_ideal = np.array(all_ideal)
        all_actual = np.array(all_actual)
        errors = all_actual - all_ideal
        rmse = np.sqrt(np.mean(errors ** 2))
        nrmse_pct = (rmse / 255 / 15 / 4) * 100
        sig_pwr = np.mean(all_ideal ** 2)
        noise_pwr = np.mean(errors ** 2)
        snr = 10 * np.log10(sig_pwr / noise_pwr) if noise_pwr > 0 else float('inf')

        print(f"测试完成！")
        print(f"归一化均方根误差 (NRMSE): {nrmse_pct:.6f}%")
        print(f"SNR: {snr:.6f}dB")
        print(f"{'=' * 60}")

        return all_ideal, all_actual


# --- 主程序 ---
if __name__ == "__main__":
    # 实例化处理器
    processor = Test_RRAM_VMM_Processor()

    # 运行批量测试
    ideal_data, actual_data = processor.batch_test(1000)

    # --- 可视化结果 ---
    plt.figure(figsize=(12, 5))

    # 图1: 散点对比图
    plt.subplot(1, 2, 1)
    plt.scatter(ideal_data, actual_data, alpha=0.5, s=10, c='blue', label='Data Points')
    # 画一条 y=x 的红色参考线
    plt.plot([0, 4], [0, 4], 'r--', linewidth=2, label='Ideal (y=x)')
    plt.title('Accuracy Check: Ideal vs. RRAM Simulated')
    plt.xlabel('Ideal Math Result')
    plt.ylabel('RRAM Simulator Result')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 图2: 误差直方图
    plt.subplot(1, 2, 2)
    error_values = (actual_data - ideal_data) / 255 / 15 / 4 * 100
    plt.hist(error_values, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title('Error Distribution (Simulation - Ideal)')
    plt.xlabel('Error Value %')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()