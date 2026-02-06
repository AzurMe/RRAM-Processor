import numpy as np
import matplotlib.pyplot as plt


class Differential_RRAM_Processor:
    def __init__(self):
        # --- 1. 系统参数 ---
        self.vdd = 1.8
        self.v_cm = 0.9  # 共模电压，代表逻辑 0
        self.adc_bits = 8
        self.dac_bits = 8
        self.array_size = (4, 4)

        # --- 2. RRAM 参数 (4-bit Cell) ---
        self.r_lrs = 10e3
        self.r_hrs = 1e6
        self.g_unit_lrs = 1 / self.r_lrs
        self.g_unit_hrs = 1 / self.r_hrs

        # --- 3. 差分 TIA 设计 ---
        # 最大单边电流：4行 * 1.8V * 15单位LRS
        max_weight_factor = 15
        max_current_single_side = self.array_size[0] * (self.vdd * max_weight_factor * self.g_unit_lrs)

        # 设定 Rf，使最大差分电流产生的电压摆幅为 0.85V (相对于 Vcm)
        self.rf = 0.85 / max_current_single_side

        # --- 4. LUT 校准 (支持 -15300 到 +15300 的逻辑值) ---
        self.lut = self._calibrate_differential_lut()

    def _quantize(self, val, bits, v_min, v_max):
        levels = 2 ** bits - 1
        v_range = v_max - v_min
        normalized = (val - v_min) / v_range
        code = np.round(normalized * levels)
        code = np.clip(code, 0, levels)
        quantized_val = v_min + (code / levels) * v_range
        return quantized_val, code.astype(int)

    def _calibrate_differential_lut(self):
        """映射 ADC Code -> 逻辑结果 (-15300 ~ 15300)"""
        lut = {}
        # 现在的满量程逻辑值：4行 * 255输入 * 15权重
        logic_max = self.array_size[0] * 255 * 15  # 15300

        for code in range(256):
            v_sim = (code / 255) * self.vdd
            # 0.9V 是我们在 Rf 计算时设定的目标单边最大摆幅 (0.85V) 的理论参考
            # 逻辑值 = ((当前电压 - 共模电压) / 最大摆幅) * 最大逻辑值
            logic_val = ((v_sim - self.v_cm) / 0.85) * logic_max
            lut[code] = logic_val
        return lut

    def _weight_to_conductance(self, w_int):
        """将 -15~15 的整数权重映射为差分电导对"""
        g_plus = np.zeros_like(w_int, dtype=float)
        g_minus = np.zeros_like(w_int, dtype=float)

        for i in range(w_int.shape[0]):
            for j in range(w_int.shape[1]):
                val = abs(int(w_int[i, j]))
                g_val = 0
                for b in range(4):
                    factor = 2 ** b
                    if (val >> b) & 1:
                        g_val += factor * self.g_unit_lrs
                    else:
                        g_val += factor * self.g_unit_hrs

                if w_int[i, j] >= 0:
                    g_plus[i, j] = g_val
                    g_minus[i, j] = 15 * self.g_unit_hrs  # 负端漏电
                else:
                    g_plus[i, j] = 15 * self.g_unit_hrs  # 正端漏电
                    g_minus[i, j] = g_val
        return g_plus, g_minus

    def run(self, input_vec_255, weight_mat_15):
        # 1. DAC: 0-255 映射到 0-1.8V
        v_in = input_vec_255 * self.vdd / 255

        # 2. Weight Mapping (使用修改后的整数映射)
        g_p, g_m = self._weight_to_conductance(weight_mat_15)

        # 3. Array Compute
        i_p = np.dot(v_in, g_p)
        i_m = np.dot(v_in, g_m)

        # 4. Differential TIA: 输出以 Vcm (0.9V) 为中心摆动
        v_diff_out = self.v_cm + (i_p - i_m) * self.rf
        v_diff_out = np.clip(v_diff_out, 0, self.vdd)

        # 5. ADC
        _, adc_codes = self._quantize(v_diff_out, self.adc_bits, 0, self.vdd)

        # 6. LUT
        return np.array([self.lut[c] for c in adc_codes])
