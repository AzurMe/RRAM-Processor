import numpy as np

class RRAM_VMM_Processor:
    def __init__(self):
        # --- 1. System Parameter Definition ---
        self.vdd = 1.8  # Supply Voltage (V)
        self.vread = 0.3
        self.v_min = 0.1  # Leave a 100mV margin at the bottom
        self.v_max = 1.7  # Leave a 100mV margin at the top
        self.adc_bits = 8  # ADC bit width
        self.dac_bits = 8  # DAC bit width
        self.array_size = (4, 4)  # 4x4 array
        self.cell_bit = 4

        # --- 2. Device Physical Parameters (Typical Values) ---
        # Setting the RRAM resistance value
        # LRS (Low Resistance State, Logic 1): 10 kOhm
        # HRS (High Resistance State, Logic 0): 1 MOhm (Assuming a switching ratio of 100)
        self.r_lrs = 10e3
        self.r_hrs = 1e6

        # Conductance
        self.g_lrs = 1 / self.r_lrs
        self.g_hrs = 1 / self.r_hrs

        # --- 3. TIA  ---
        self.i_max = self.array_size[0] * (self.vread * self.g_lrs)
        self.i_min = self.array_size[0] * (self.vread * self.g_hrs)
        self.rf = (self.v_max - self.v_min) / (self.i_max - self.i_min)
        # print(f"[Init] Rf set to: {self.rf:.2f} Ohms")

        # --- 4.  LUT ---

        self.calibration_value = 2
        self.lut = self.calibrate_lut()

    def calibrate_lut(self):
        lut = {}
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
        input_vector = np.clip(input_vector, 0, 256)
        analog_voltages = input_vector * self.vread / 255
        return analog_voltages

    def rram_array_processing(self, voltage_vector, weight_matrix):
        conductance_matrix = np.where(weight_matrix == 1, self.g_lrs, self.g_hrs)
        current_vector = np.dot(voltage_vector, conductance_matrix)
        return current_vector

    def tia(self, current_vector):
        tia_voltages = (current_vector) * self.rf
        return tia_voltages

    def adc(self, tia_voltages):
        level = 2**self.adc_bits - 1
        digital_codes = np.round(tia_voltages / self.vdd * level)
        return digital_codes.astype(int)

    def lut_mapping(self, digital_codes):
        results = [self.lut[code] for code in digital_codes]
        return (np.array(results))

    def run(self, input_vec, weight_mat, verbose):
        if verbose:
            print("-" * 50)
            print(f"Input Vector (Logic): {input_vec}")
            print(f"Weight Matrix (Logic):\n{weight_mat}")

        # 1. DAC
        v_in = self.dac(input_vec)
        if verbose:
            print(f"1. DAC Output (Volts): {np.round(v_in, 3)}")

        # 2. RRAM Array (Analog Compute)
        i_out = self.rram_array_processing(v_in, weight_mat)
        if verbose:
            print(f"2. Array Bitline Current (uA): {np.round(i_out * 1e6, 1)}")

        # 3. TIA
        tia_voltage = self.tia(i_out)
        if verbose:
            print(f"3. TIA output voltage (V): {tia_voltage}")
        # 4. ADC
        adc_codes = self.adc(tia_voltage)
        if verbose:
            print(f"4. ADC Output Codes (0-255): {adc_codes}")

        # 5. LUT Mapping
        final_result = self.lut_mapping(adc_codes)
        if verbose:
            print(f"5. Final Result (via LUT): {final_result}")

        # 6. Theoretical Check (Ideal Math)
        ideal_result = np.dot(input_vec, weight_mat)
        if verbose:
            print(f"6. Ideal Math Result: {ideal_result}")

        return final_result


