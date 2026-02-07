import numpy as np
import matplotlib.pyplot as plt
from Model.ideal_model import RRAM_VMM_Processor

class Test_RRAM_VMM_Processor:
    def batch_test(self, num_samples=100, verbose=0):
        """
        Randomly generate test sets and analyze errors
        """
        print(f"\n{'=' * 20} Start batch testing (N={num_samples}) {'=' * 20}")

        all_ideal = []
        all_actual = []
        errors = []
        rmse_all = []
        for i in range(num_samples):
            # 1. Randomly Generated Inputs
            vec = np.random.randint(0,256, size=(4))
            # 2. Randomly Generated Weights
            mat = np.random.randint(0, 2, size=(4, 4))

            # 3. Calculate the ideal value
            ideal_res = np.dot(vec, mat)

            # 4. Calculated simulated value
            processor_instance = RRAM_VMM_Processor()
            sim_res = processor_instance.run(vec, mat,verbose)

            # 5. Record data
            all_ideal.extend(ideal_res)
            all_actual.extend(sim_res)

        all_ideal = np.array(all_ideal)
        all_actual = np.array(all_actual)
        errors = all_actual - all_ideal
        rmse = np.sqrt(np.mean(errors ** 2))
        nrmse_pct = (rmse / 255 / 4) * 100
        sig_pwr = np.mean(all_ideal**2)
        noise_pwr = np.mean(errors**2)
        snr = 10 * np.log10(sig_pwr / noise_pwr) if noise_pwr > 0 else float('inf')

        print(f"Test completed！")
        print(f"Normalized Root Mean Square Error (NRMSE): {nrmse_pct:.6f}%")
        print(f"SNR: {snr:.6f}dB")
        print(f"{'=' * 60}")

        return all_ideal, all_actual


# --- 主程序 ---
if __name__ == "__main__":
    processor = Test_RRAM_VMM_Processor()

    ideal_data, actual_data = processor.batch_test(1000,0)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(ideal_data, actual_data, alpha=0.5, s=10, c='blue', label='Data Points')

    plt.plot([0, 1040], [0, 1040], 'r--', linewidth=2, label='Ideal (y=x)')
    plt.title('Accuracy Check: Ideal vs. RRAM Simulated')
    plt.xlabel('Ideal Math Result')
    plt.ylabel('RRAM Simulator Result')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    error_values = (actual_data - ideal_data) / 255 / 4
    plt.hist(error_values, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title('Error Distribution (Simulation - Ideal)')
    plt.xlabel('Error Value %')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

