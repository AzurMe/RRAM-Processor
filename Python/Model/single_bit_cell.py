import numpy as np


class RRAM_VMM_Processor:
    def __init__(self):
        # --- 1. System Parameter Definition ---
        self.vdd = 1.8  # Supply Voltage (V)
        self.vread = 0.3  # Target read voltage drop across RRAM cells

        # TIA Parameters: Based on OTA's Input Common Mode Range (ICMR)
        self.v_cm_tia = 0.5  # OTA non-inverting input level (V_ref/Common-mode)
        self.v_max_tia = 1.6  # TIA output upper limit (V)
        self.v_min_tia = 0.5  # TIA output starting point (at zero current)

        # ADC Parameters: Differential input 0.2V to 1.6V
        # V_diff = V_tia_out - V_adc_low
        self.v_adc_low = 0.2  # ADC inverting input connected to 0.2V
        self.v_adc_range = 1.6  # ADC full-scale differential range (V)
        self.adc_bits = 8

        self.array_size = (4, 4)  # 4x4 RRAM Crossbar
        self.cell_bit = 4  # 4-bit weights per cell (0-15)

        # --- 2. Device Physical Parameters ---
        self.r_lrs = 10e3  # Low Resistance State (10k Ohm)
        self.r_hrs = 1e6  # High Resistance State (1M Ohm)
        self.g_lrs = 1 / self.r_lrs
        self.g_hrs = 1 / self.r_hrs

        # --- 3. Feedback Resistor (Rf) Calculation ---
        # Calculate max current when all rows are 255 and all weights are LRS
        self.i_max = self.array_size[0] * (self.vread * self.g_lrs)

        # Rf = Effective Voltage Swing / Max Current
        # Effective Swing = 1.6V (V_max) - 0.5V (V_start) = 1.1V
        self.rf = (self.v_max_tia - self.v_min_tia) / self.i_max

        # --- 4. LUT Calibration ---
        self.lut = self.calibrate_lut()

    def calibrate_lut(self):
        """
        Calibrate Look-Up Table (LUT):
        Maps ADC digital codes to logical VMM results (Input * Weight sum).
        Compensates for TIA offset (0.5V) and ADC negative terminal offset (0.2V).
        """
        lut = {}
        # Calculate the ADC code increment produced by one unit of (Input=1 * Weight=1)
        i_unit = (1 / 255 * self.vread) * self.g_lrs
        v_diff_unit = i_unit * self.rf
        code_per_unit = (v_diff_unit / self.v_adc_range) * (2 ** self.adc_bits - 1)

        for code in range(2 ** self.adc_bits):
            # Convert ADC code back to physical differential voltage
            v_diff_actual = (code / (2 ** self.adc_bits - 1)) * self.v_adc_range

            # Physical offset compensation:
            # Pure signal = V_diff_measured - (V_tia_idle - V_adc_low)
            # Idle V_diff = 0.5V - 0.2V = 0.3V
            v_diff_pure_signal = v_diff_actual - (self.v_cm_tia - self.v_adc_low)

            logic_val = v_diff_pure_signal / v_diff_unit
            lut[code] = max(0, int(np.round(logic_val)))
        return lut

    def dac(self, input_vector):
        """
        DAC Output: Shifts the logical voltage by 0.5V (Common-mode).
        Ensures RRAM voltage drop = (V_dac - 0.5V) matches intended logic.
        """
        input_vector = np.clip(input_vector, 0, 255)
        # Shifted range: 0.5V (Logic 0) to 0.8V (Logic 255)
        analog_voltages = (input_vector * self.vread / 255) + self.v_cm_tia
        return analog_voltages

    def rram_array_processing(self, voltage_vector, weight_matrix):
        """
        Analog computation in RRAM Array.
        I_BL = (V_row - V_column_ref) * Conductance
        """
        # Linear mapping of weights to conductance
        # In a real 4-bit cell, this would be a multi-level conductance mapping
        conductance_matrix = np.where(weight_matrix == 1, self.g_lrs, self.g_hrs)

        # Effective voltage drop across the resistor
        v_effective = voltage_vector - self.v_cm_tia
        current_vector = np.dot(v_effective, conductance_matrix)
        return current_vector

    def tia(self, current_vector):
        """
        Transimpedance Amplifier: V_out = V_ref + I * Rf
        Converts Bitline current back to voltage starting from 0.5V.
        """
        tia_voltages = self.v_cm_tia + (current_vector * self.rf)
        # Physical saturation clipping at VDD
        return np.clip(tia_voltages, 0, self.vdd)

    def adc(self, tia_voltages):
        """
        Differential SAR ADC:
        Samples the difference between TIA output and 0.2V reference.
        """
        # V_diff = V_tia_out (0.5V-1.6V) - V_adc_low (0.2V)
        v_diff = tia_voltages - self.v_adc_low
        v_diff = np.clip(v_diff, 0, self.v_adc_range)

        level = 2 ** self.adc_bits - 1
        digital_codes = np.round(v_diff / self.v_adc_range * level)
        return digital_codes.astype(int)

    def lut_mapping(self, digital_codes):
        """Map ADC codes to final logical integers using the calibrated LUT."""
        return np.array([self.lut[code] for code in digital_codes])

    def run(self, input_vec, weight_mat, verbose=False):
        """Main execution flow for the VMM processor."""
        # 1. DAC Phase
        v_in = self.dac(input_vec)

        # 2. RRAM Computing Phase
        i_out = self.rram_array_processing(v_in, weight_mat)

        # 3. TIA Phase
        v_tia = self.tia(i_out)

        # 4. ADC Phase
        adc_codes = self.adc(v_tia)

        # 5. Logical Restoration Phase
        final_result = self.lut_mapping(adc_codes)

        if verbose:
            print("-" * 50)
            print(f"TIA Output (V): {np.round(v_tia, 3)}")
            print(f"ADC Digital Codes: {adc_codes}")
            print(f"Ideal Result: {np.dot(input_vec, weight_mat)}")
            print(f"Final Result: {final_result}")

        return final_result