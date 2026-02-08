# RRAM Vector-Matrix Multiplication (VMM) Simulator

## 📌 Project Overview

This project is a Python-based simulator for **Resistive Random Access Memory (RRAM)** based In-Memory Computing (IMC). It specifically simulates the **Vector-Matrix Multiplication (VMM)** operation, which is the core computation in neural networks.

The simulator models the complete analog signal chain:

**Digital Input $\rightarrow$ DAC $\rightarrow$ RRAM Crossbar Array $\rightarrow$ TIA $\rightarrow$ADC $\rightarrow$ LUT Calibration $\rightarrow$ Digital Output**

It includes two distinct models:

1. **Single-Bit Cell Model**: An ideal simulation for architectural verification using 1T1R (One Transistor One Resistor) cells.
2. **4-Bit Cell Non-Ideal Model**: A high-fidelity simulation that models multi-bit weight cells (via parallel conductance) and incorporates various circuit-level non-idealities such as device variation, thermal noise, and gain errors.

---

## ⚙️ Technical Specifications

### 1. Signal Chain Architecture

- **DAC**: 8-bit resolution. Converts digital inputs (0-255) to analog read voltages ($V_{read}$).
- **RRAM Array**: A $4 \times 4$ crossbar array performing analog Multiply-Accumulate (MAC) operations based on Kirchhoff's laws.
- **TIA (Transimpedance Amplifier)**: Converts bitline current ($I_{BL}$) to voltage ($V_{out}$). The feedback resistance ($R_f$) is auto-calibrated to map the current range to the voltage swing (0.1V - 1.7V).
- **ADC**: 8-bit resolution. Quantizes the TIA output voltage back to digital codes.
- **LUT**: A post-processing calibration module that maps ADC codes to logical integer results, effectively canceling out leakage currents (HRS offset) and static non-linearities.

### 2. Device Parameters (Typical)

- **Supply Voltage ($V_{DD}$)**: 1.8 V
- **Read Voltage ($V_{read}$)**: 0.3 V
- **LRS (Low Resistance State / Logic 1)**: $10\ k\Omega$
- **HRS (High Resistance State / Logic 0)**: $1\ M\Omega$ (On/Off Ratio $\approx$ 100)
- **Output Margins**: 100mV reserved at both top and bottom rails ($V_{min}=0.1V$, $V_{max}=1.7V$).

---

## 📉 Non-Ideal Factors (4-Bit Model)

The `four_bit_cell.py` module introduces realistic physical constraints to simulate hardware behavior accurately:

| **Parameter** | **Variable** | **Description** | **Typical Value** |
| --- | --- | --- | --- |
| **Conductance Variation** | `g_sigma` | Device-to-device variation modeled as a Gaussian distribution percentage. | 5% |
| **TIA Thermal Noise** | `v_noise` | Additive White Gaussian Noise (AWGN) introduced by the Op-Amp and feedback resistor. | 5 mV |
| **ADC Input Noise** | `n_adc` | Equivalent input noise at the ADC sampling stage. | 5 mV |
| **DAC Gain Error** | `dac_gain_err` | Linear gain mismatch in the input voltage generation. | 1% |

### 4-Bit Weight Implementation

In the 4-bit model, a single logical weight (0-15) is realized by summing the currents of **4 parallel RRAM branches** (or a multi-level cell analogy), scaled by binary weights ($2^0, 2^1, 2^2, 2^3$). Non-idealities are applied to each branch independently.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following Python libraries installed:

Bash

`pip install numpy matplotlib`

### Running the Simulations

### 1. Ideal Single-Bit Simulation

Run the test script to verify the basic logic and linearity of the architecture.

Bash

`python Test_single_bit_cell.py`

- **Output**: Displays the Normalized Root Mean Square Error (NRMSE) and generates a scatter plot comparing Ideal vs. Simulated results.

### 2. Non-Ideal 4-Bit Simulation

Run this script to perform a batch Monte Carlo simulation (Default N=1000 samples). This evaluates the robustness of the system against noise.

Bash

`python Test_four_bit_cell.py`

- **Output**:
    - Prints configuration of non-ideal parameters.
    - Calculates **SNR (Signal-to-Noise Ratio)** in dB.
    - Generates error distribution histograms.

---

## 📊 Visualization

The test scripts generate two primary plots using `matplotlib`:

1. **Accuracy Check (Left Plot)**:
    - **X-Axis**: Ideal mathematical result (`np.dot`).
    - **Y-Axis**: RRAM Simulator result.
    - **Red Dashed Line**: The ideal y=x reference. The tighter the data points cluster around this line, the higher the precision.
2. **Error Distribution (Right Plot)**:
    - A histogram showing the percentage error between the simulated hardware output and the ideal math value.
    - Helps in analyzing whether the error is systematic (offset) or random (Gaussian).