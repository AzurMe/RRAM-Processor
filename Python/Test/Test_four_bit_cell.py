import numpy as np
import matplotlib.pyplot as plt
from Python.Model.four_bit_cell import RRAM_VMM_Processor_4b_nonideal

class Test_RRAM_VMM_Processor:
    def batch_test(self, num_samples=100, verbose=0, g_sigma=0.05, v_noise=0.005, dac_gain_err = 0.01, n_adc = 0.005):
        """
        Randomly generate test sets and analyze errors
        """
        print(f"\n{'=' * 20} Start batch testing (N={num_samples}) {'=' * 20}")

        all_ideal = []
        all_actual = []
        errors = []

        for i in range(num_samples):
            # 1. Randomly Generated Inputs
            vec = np.random.randint(0, 256, 4)
            # 2. Randomly Generated Weights
            mat = np.random.randint(0, 16, size=(4,4))

            # 3. Calculate the ideal value
            ideal_res = np.dot(vec, mat)

            # 4. Calculated simulated value
            processor_instance = RRAM_VMM_Processor_4b_nonideal(g_sigma, v_noise, dac_gain_err, n_adc)
            sim_res = processor_instance.run(vec, mat, verbose)

            # 5. Record data
            all_ideal.extend(ideal_res)
            all_actual.extend(sim_res)

        all_ideal = np.array(all_ideal)
        all_actual = np.array(all_actual)
        errors = all_actual - all_ideal
        rmse = np.sqrt(np.mean(errors ** 2))
        nrmse_pct = (rmse / 255 / 15 / 4) * 100
        sig_pwr = np.mean(all_ideal ** 2)
        noise_pwr = np.mean(errors ** 2)
        snr = 10 * np.log10(sig_pwr / noise_pwr) if noise_pwr > 0 else float('inf')

        print(f"Test completed！")
        print(f"Normalized Root Mean Square Error (NRMSE): {nrmse_pct:.6f}%")
        print(f"SNR: {snr:.6f}dB")
        print(f"{'=' * 60}")

        return all_ideal, all_actual


# --- 主程序 ---
if __name__ == "__main__":
    # Instantiated Processors
    processor = Test_RRAM_VMM_Processor()

    # Set the nonideal parameters
    g_sigma = 0.05
    v_noise = 0.005
    dac_gain_err = 0.01
    n_adc = 0.005

    print("=" * 30 + " nonideal parameters " + "=" * 30)
    print(f"Conductance standard deviation (g_sigma)：{g_sigma:.4f} S")
    print(f"TIA thermal noise (v_noise)：{v_noise:.4f} V")  # 标注单位，贴合电路场景
    print(f"DAC gain error (dac_gain_err)：{dac_gain_err:.4f}")
    print(f"ADC input refferd noise (n_adc)：{n_adc:.4f} V")
    print("=" * 70)

    ideal_data, actual_data = processor.batch_test(1000,0,g_sigma, v_noise, dac_gain_err, n_adc)

    plt.figure(figsize=(12, 5))

    #Figure 1: Scatter Comparison Chart
    plt.subplot(1, 2, 1)
    plt.scatter(ideal_data, actual_data, alpha=0.5, s=10, c='blue', label='Data Points')
    plt.plot([0, 12000], [0, 12000], 'r--', linewidth=2, label='Ideal (y=x)')
    plt.title('Accuracy Check: Ideal vs. RRAM Simulated')
    plt.xlabel('Ideal Result')
    plt.ylabel('Simulated Result')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Figure2: Error Histogram
    plt.subplot(1, 2, 2)
    error_values = (actual_data - ideal_data) / 255 / 15 / 4 * 100
    plt.hist(error_values, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title('Error Distribution (Simulation - Ideal)')
    plt.xlabel('Error Value %')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()