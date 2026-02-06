import numpy as np


class Realistic_RRAM_Processor:
    def __init__(self, g_sigma=0.05, v_noise_std=0.01):
        # --- 1. 系统参数 ---
        self.vdd = 1.8
        self.v_cm = 0.9
        self.adc_bits = 8
        self.dac_bits = 8
        self.array_size = (4, 4)

        # --- 2. 非理想性参数 ---
        self.g_sigma = g_sigma
        self.v_noise_std = v_noise_std

        # --- 3. RRAM 参数 ---
        self.r_lrs = 10e3
        self.r_hrs = 1e6
        self.g_unit_lrs = 1 / self.r_lrs
        self.g_unit_hrs = 1 / self.r_hrs

        # --- 4. 差分 TIA 设计 ---
        # 4行 * 15(最大权重因子) * 1.8V(最大输入) * G_lrs
        max_weight_factor = 15
        max_col_current = self.array_size[0] * (self.vdd * max_weight_factor * self.g_unit_lrs)

        # 设定 Rf，使最大差分摆幅为 0.8V (相对于 Vcm)
        self.rf = 0.8 / max_col_current

        # 预计算校准 LUT
        self.lut = self._calibrate_lut()

    def _quantize(self, val, bits, v_min, v_max):
        levels = 2 ** bits - 1
        normalized = (val - v_min) / (v_max - v_min)
        code = np.round(normalized * levels)
        code = np.clip(code, 0, levels)
        return v_min + (code / levels) * (v_max - v_min), code.astype(int)

    def _calibrate_lut(self):
        """
        不使用增强型 LUT，采用基于 Rf 的理论映射。
        逻辑量程范围：4行 * 255输入 * 15权重 = 15300
        """
        lut = {}
        # 理论最大逻辑点积结果
        max_logic_range = self.array_size[0] * 255 * 15

        for code in range(256):
            v_sim = (code / 255) * self.vdd
            # 这里的 0.8 必须与 __init__ 中的 Rf 计算系数严格一致
            # 公式：逻辑值 = ((当前电压 - 中间电压) / 设计的最大摆幅) * 最大逻辑范围
            logic_val = ((v_sim - self.v_cm) / 0.8) * max_logic_range
            lut[code] = logic_val
        return lut

    def run(self, input_vec_255, weight_mat_15):
        # 1. DAC: 0-255 映射到 0-1.8V
        # 注意：这里不再乘以 self.vdd，因为输入本身就是数字量
        v_in_q, _ = self._quantize(input_vec_255, self.dac_bits, 0, 255)
        v_in = (v_in_q / 255) * self.vdd

        # 2. 权重映射并加入波动
        # 权重已经是 0-15 整数，无需 round(weight_mat * 15)
        g_p = np.zeros_like(weight_mat_15, dtype=float)
        g_m = np.zeros_like(weight_mat_15, dtype=float)

        for i in range(self.array_size[0]):
            for j in range(self.array_size[1]):
                val = int(abs(weight_mat_15[i, j]))
                g_val = 0
                for b in range(4):
                    factor = 2 ** b
                    # 4-bit 单元内部 4 个器件独立噪声
                    is_lrs = (val >> b) & 1
                    unit_g = self.g_unit_lrs if is_lrs else self.g_unit_hrs
                    g_val += factor * unit_g * np.random.normal(1.0, self.g_sigma)

                if weight_mat_15[i, j] >= 0:
                    g_p[i, j] = g_val
                    # 负端保持全 HRS 噪声
                    g_m[i, j] = 15 * self.g_unit_hrs * np.random.normal(1.0, self.g_sigma)
                else:
                    g_p[i, j] = 15 * self.g_unit_hrs * np.random.normal(1.0, self.g_sigma)
                    g_m[i, j] = g_val

        # 3. 阵列计算
        i_p = np.dot(v_in, g_p)
        i_m = np.dot(v_in, g_m)

        # 4. TIA 阶段
        v_diff_signal = (i_p - i_m) * self.rf
        v_tia_out = self.v_cm + v_diff_signal
        # 叠加电路热噪声
        v_tia_noisy = v_tia_out + np.random.normal(0, self.v_noise_std, v_tia_out.shape)

        # 5. ADC 阶段
        v_tia_noisy = np.clip(v_tia_noisy, 0, self.vdd)
        _, adc_codes = self._quantize(v_tia_noisy, self.adc_bits, 0, self.vdd)

        # 6. LUT 映射
        return np.array([self.lut[c] for c in adc_codes])
