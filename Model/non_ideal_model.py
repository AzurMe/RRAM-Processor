import numpy as np

class RRAM_VMM_Processor_4b_nonideal:
    def __init__(self, g_sigma=0.05, v_noise=0.005, dac_gain_err = 0.01, n_adc = 0.005):
        # --- 1. 系统参数定义 ---
        self.vdd = 1.8  # 电源电压 (V)
        self.vread = 0.3
        self.v_min = 0.1  # 底部留出 100mV 裕量
        self.v_max = 1.7  # 顶部留出 100mV 裕量
        self.adc_bits = 8  # ADC位宽
        self.dac_bits = 8  # DAC位宽
        self.array_size = (4, 4)  # 4x4 阵列
        self.cell_bit = 4

        # --- 2. 器件物理参数 (典型值) ---
        # 设定 RRAM 电阻值
        # LRS (Low Resistance State, 逻辑1): 10 kOhm
        # HRS (High Resistance State, 逻辑0): 1 MOhm (假设开关比为100)
        self.r_lrs = 10e3
        self.r_hrs = 1e6

        # 对应的电导值 (Conductance)
        self.g_lrs = 1 / self.r_lrs
        self.g_hrs = 1 / self.r_hrs

        # --- 3. TIA (跨阻放大器) 设计 ---
        self.i_max = self.array_size[0] * (self.vread * self.g_lrs) * ((2 ** self.cell_bit) - 1)
        self.i_min = self.array_size[0] * (self.vread * self.g_hrs) * ((2 ** self.cell_bit) - 1)
        self.rf = (self.v_max - self.v_min) / (self.i_max - self.i_min)
        print(f"[Init] Rf set to: {self.rf:.2f} Ohms")

        # --- 4. 生成 LUT (查找表) ---
        # 在实际芯片中，LUT用于消除HRS漏电和非线性误差，将ADC码值映射回逻辑结果
        self.calibration_value = 8
        self.lut = self.calibrate_lut()

        # --- 非理想性参数 ---
        self.g_sigma = g_sigma  # 电导波动标准差 (e.g., 5%)
        self.v_noise = v_noise  # TIA 热噪声标准差 (e.g., 5mV)
        self.n_adc = n_adc
        self.dac_gain_err = dac_gain_err  # DAC 增益误差 (1%)

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
        gain_error = 1 + np.random.uniform(-self.dac_gain_err, self.dac_gain_err)
        input_vector = np.clip(input_vector, 0, 256)
        analog_voltages = (input_vector * self.vread / 255) * gain_error
        return analog_voltages

    def rram_array_processing(self, voltage_vector, weight_matrix):
        """计算 4-bit 权重产生的等效电导 (含 4 个并联单元)"""
        rows, cols = weight_matrix.shape
        g_matrix = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                val = int(weight_matrix[r, c])
                g_cell = 0
                for bit_pos in range(4):
                    weight_factor = 2 ** bit_pos
                    bit_val = (val >> bit_pos) & 1
                    # 模拟支路并联：1 对应 LRS，0 对应 HRS
                    cond = self.g_lrs if bit_val == 1 else self.g_hrs
                    cond = cond + cond * np.random.normal(0,self.g_sigma)
                    g_cell += weight_factor * cond
                g_matrix[r, c] = g_cell

        current_vector = np.dot(voltage_vector, g_matrix)
        return current_vector

    def tia(self, current_vector):
        tia_voltages = current_vector * self.rf
        #TIA噪声
        tia_voltages += np.random.normal(0, self.v_noise, tia_voltages.shape)
        return tia_voltages

    def adc(self, tia_voltages):
        level = 2 ** self.adc_bits - 1
        digital_codes = np.round((tia_voltages + np.random.normal(0,self.n_adc))  / self.vdd * level)
        return digital_codes.astype(int)

    def lut_mapping(self, digital_codes):
        results = [self.lut[code] for code in digital_codes]
        return (np.array(results))

    def run(self, input_vec, weight_mat):
        print("-" * 50)
        print(f"Input Vector (Logic): {input_vec}")
        print(f"Weight Matrix (Logic):\n{weight_mat}")

        # 1. DAC
        v_in = self.dac(input_vec)
        print(f"1. DAC Output (Volts): {np.round(v_in, 3)}")

        # 2. RRAM Array (Analog Compute)
        i_out = self.rram_array_processing(v_in, weight_mat)
        print(f"2. Array Bitline Current (uA): {np.round(i_out * 1e6, 1)}")

        # 3. TIA
        tia_voltage = self.tia(i_out)

        # 4. ADC
        adc_codes = self.adc(tia_voltage)
        print(f"3. ADC Output Codes (0-255): {adc_codes}")

        # 5. LUT Mapping
        final_result = self.lut_mapping(adc_codes)
        # final_result = final_result - self.calibration_value
        print(f"4. Final Result (via LUT): {final_result}")

        # 6. Theoretical Check (Ideal Math)
        ideal_result = np.dot(input_vec, weight_mat)
        print(f"5. Ideal Math Result: {ideal_result}")

        return final_result
