import numpy as np


class RRAM_VMM_Processor_4b_nonideal:
    def __init__(self, g_sigma=0.05, v_noise=0.005):
        # --- 1. 系统参数 ---
        self.vdd, self.vread = 1.8, 0.3
        self.v_min, self.v_max = 0.1, 1.7
        self.adc_bits, self.array_size = 8, (4, 4)
        self.cell_bit = 4

        # --- 非理想性参数 ---
        self.g_sigma = g_sigma  # 电导波动标准差 (e.g., 5%)
        self.v_noise = v_noise  # TIA 热噪声标准差 (e.g., 5mV)
        self.dac_gain_err = 0.01  # DAC 增益误差 (1%)

        # --- 2. 器件参数 ---
        self.r_lrs, self.r_hrs = 10e3, 1e6
        self.g_lrs, self.g_hrs = 1 / self.r_lrs, 1 / self.r_hrs

        # --- 3. TIA 设计 ---
        # 满载电流: 4行 * 0.3V * (15*g_lrs)
        self.i_max = self.array_size[0] * (self.vread * 15 * self.g_lrs)
        self.i_min = self.array_size[0] * (self.vread * 15 * self.g_hrs)
        self.rf = (self.v_max - self.v_min) / (self.i_max - self.i_min)

        # --- 4. 生成 LUT ---
        self.calibration_value = 8
        self.lut = self.calibrate_lut()

    def calibrate_lut(self):
        lut = {}
        # 计算：当输入为 1 (数字量)，权重为 1 (LRS) 时，产生的电压贡献
        v_unit_input = (1 / 255) * self.vread
        v_logic_1 = v_unit_input * self.g_lrs * self.rf

        for code in range(256):
            voltage = (code / 255) * self.vdd
            logic_val = voltage / v_logic_1
            lut[code] = int(np.round(logic_val, 2) - self.calibration_value)
            if lut[code] < 0:
                lut[code] = 0
        return lut

    def dac(self, input_vector):
        """引入 DAC 增益误差"""
        gain_error = 1 + np.random.uniform(-self.dac_gain_err, self.dac_gain_err)
        analog_voltages = (input_vector * self.vread / 255) * gain_error
        return np.clip(analog_voltages, 0, self.vread)

    def rram_array_processing(self, voltage_vector, weight_matrix):
        """计算含电导波动的 4-bit 权重电导"""
        rows, cols = weight_matrix.shape
        g_matrix = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                val = int(weight_matrix[r, c])
                g_cell_total = 0
                for bit_pos in range(4):
                    weight_factor = 2 ** bit_pos
                    bit_val = (val >> bit_pos) & 1
                    ideal_cond = self.g_lrs if bit_val == 1 else self.g_hrs

                    # --- 引入非理想性：电导波动 ---
                    real_cond = ideal_cond * (1 + np.random.normal(0, self.g_sigma))
                    g_cell_total += weight_factor * real_cond
                g_matrix[r, c] = g_cell_total

        return np.dot(voltage_vector, g_matrix)

    def tia(self, current_vector):
        """引入 TIA 偏置与热噪声"""
        v_tia = (current_vector * self.rf) + self.v_min
        # 加入高斯噪声
        v_tia += np.random.normal(0, self.v_noise, v_tia.shape)
        return v_tia

    def adc(self, tia_voltages):
        """ADC 量化"""
        v_clip = np.clip(tia_voltages, 0, self.vdd)
        return np.round(v_clip / self.vdd * 255).astype(int)

    def run(self, input_vec, weight_mat):
        v_in = self.dac(input_vec)
        i_out = self.rram_array_processing(v_in, weight_mat)
        v_tia = self.tia(i_out)
        adc_codes = self.adc(v_tia)
        final_result = np.array([self.lut[code] for code in adc_codes])


        return final_result
