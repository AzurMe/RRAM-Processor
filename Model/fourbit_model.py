import numpy as np
import matplotlib.pyplot as plt


class RRAM_4Bit_VMM_Processor:
    def __init__(self):
        # --- 1. 系统参数 ---
        self.vdd = 1.8
        self.adc_bits = 8
        self.dac_bits = 8
        self.array_size = (4, 4)

        # --- 2. RRAM 器件参数 ---
        self.r_lrs = 10e3
        self.r_hrs = 1e6
        self.g_lrs_unit = 1 / self.r_lrs
        self.g_hrs_unit = 1 / self.r_hrs

        # --- 3. TIA 设计 (适配 255输入与15权重) ---
        # 物理最大电流逻辑：4行 x (1.8V电压) x (15倍单元电导)
        max_weight_factor = 15
        max_col_conductance = self.array_size[0] * (max_weight_factor * self.g_lrs_unit)
        max_col_current = self.vdd * max_col_conductance

        # 设置 Rf 以防止 ADC 输入溢出 (95% 安全裕量)
        self.rf = (self.vdd * 0.95) / max_col_current

        print(f"[System Init] Input: 0-255, Weight: 0-15")
        print(f"[System Init] TIA Rf: {self.rf:.2f} Ohms")

        # --- 4. 生成校准 LUT ---
        self.lut = self._calibrate_lut()

    def _quantize(self, value, bits, v_ref):
        levels = 2 ** bits - 1
        code = np.round((value / v_ref) * levels)
        code = np.clip(code, 0, levels)
        quantized_val = (code / levels) * v_ref
        return quantized_val, code.astype(int)

    def _calibrate_lut(self):
        """
        修正校准逻辑：
        最大点积结果 = 4行 * 255(输入) * 15(权重) = 15300
        """
        lut = {}
        # 满量程对应的物理电压点对应的逻辑点积值
        max_logic_val = self.array_size[0] * 255 * 15

        for code in range(256):
            voltage = (code / 255) * self.vdd
            # 这里的 0.95 是 Rf 计算时的缩放因子
            logic_val = (voltage / (self.vdd * 0.95)) * max_logic_val
            lut[code] = logic_val
        return lut

    def calculate_cell_conductance(self, w_int_matrix):
        """计算 4-bit 权重产生的等效电导 (含 4 个并联单元)"""
        rows, cols = w_int_matrix.shape
        g_matrix = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                val = int(w_int_matrix[r, c])
                g_cell = 0
                for bit_pos in range(4):
                    weight_factor = 2 ** bit_pos
                    bit_val = (val >> bit_pos) & 1
                    # 模拟支路并联：1 对应 LRS，0 对应 HRS
                    cond = self.g_lrs_unit if bit_val == 1 else self.g_hrs_unit
                    g_cell += weight_factor * cond
                g_matrix[r, c] = g_cell
        return g_matrix

    def run(self, input_vec_int, weight_mat_int):
        # 1. DAC: 0-255 输入映射到 0-1.8V
        v_in_analog = input_vec_int * self.vdd / 255

        # 2. 权重物理电导映射
        g_matrix = self.calculate_cell_conductance(weight_mat_int)

        # 3. 物理计算 (KCL)
        i_out = np.dot(v_in_analog, g_matrix)

        # 4. TIA & ADC (0-255 code)
        v_tia = np.clip(i_out * self.rf, 0, self.vdd)
        _, adc_codes = self._quantize(v_tia, self.adc_bits, self.vdd)

        # 5. LUT 映射回逻辑结果
        return np.array([self.lut[c] for c in adc_codes])

