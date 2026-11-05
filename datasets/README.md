# Hardware Trojan Detection & Evaluation Datasets

This repository consolidates four primary datasets for Hardware Trojan (HT) detection, localization, side-channel analysis, and logic synthesis benchmarking:

---

## 1. Trust-Hub Hardware Trojan Benchmarks
* **Path**: [`datasets/trust_hub_benchmarks/`](file:///home/revinr/Soykot/HT_Detection/datasets/trust_hub_benchmarks)
* **Primary Source**: [Trust-Hub](https://trust-hub.org/#/benchmarks/chip-level-trojan)
* **Contents**:
  * **`extracted/`**: Fully unpacked flagship benchmarks (`AES-T100` to `AES-T2800`), each containing Verilog source files, testbenches, simulation do-files, and technical documentation PDFs detailing the Trojan activation trigger and payload mechanism.
  * **`archives/`**: 119 original `.zip` and `.rar` packages covering 15 design families:
    * Cryptographic: `AES`, `BasicRSA`
    * Communications & Interfaces: `RS232`, `EthernetMAC10GE`, `wb_conmax` (Wishbone), `vga_lcd`
    * Processors & Controllers: `MC8051`, `PIC16F84`, `memctrl`
    * Sequential / ITC / ISCAS: `b19`, `s15850`, `s35932`, `s38417`, `s38584`, `MultPyramid`
  * **`TjFree/`**: 339 Golden (Trojan-free) reference circuits (ISCAS'85, OpenCores, ALUs, RISC-V cores, counters, arithmetic modules).

---

## 2. Hardware Trojan Power & EM Side-Channel Dataset
* **Path**: [`datasets/ieee_dataport_side_channel/`](file:///home/revinr/Soykot/HT_Detection/datasets/ieee_dataport_side_channel)
* **Primary Source**: [IEEE DataPort (DOI: 10.21227/9fwb-8978)](https://dx.doi.org/10.21227/9fwb-8978)
* **Authors**: R. Yasaei, S. Faezi, M. A. Al Faruque (UC Irvine)
* **Contents**:
  * Measured physical power and electromagnetic (EM) side-channel execution traces recorded from FPGA implementations of Trust-Hub benchmarks (`AES-T400` through `AES-T2000`) under controlled temperature (25°C).
  * Covers both **Trojan Inactive** (baseline + dormant static leakage) and **Trojan Activated** (dynamic power variations) scenarios.
  * **Download script**: `download_dataset.py` to automate downloading the 12 zip files (~2.0 GB total) with a free IEEE web account session cookie.

---

## 3. Fast & Accurate Identification of HT Locations
* **Path**: [`datasets/fast_accurate_ht_locations/`](file:///home/revinr/Soykot/HT_Detection/datasets/fast_accurate_ht_locations)
* **Primary Source**: [GitHub: aninchatto/Fast-and-Accurate-Identification-of-Hardware-Trojan-Locations-in-Gate-Level-Netlist-using-ML](https://github.com/aninchatto/Fast-and-Accurate-Identification-of-Hardware-Trojan-Locations-in-Gate-Level-Netlist-using-ML)
* **Contents**:
  * Complete repository for graph-based and nearest-neighbour ML Hardware Trojan detection and fine-grained gate-level localization.
  * Includes the full research paper PDF, evaluation models, formal verification comparisons, and raw RTL datasets (`data/TjIn`, `data/TjFree`).

---

## 4. EPFL Combinational Benchmark Suite
* **Path**: [`datasets/epfl_benchmarks/`](file:///home/revinr/Soykot/HT_Detection/datasets/epfl_benchmarks)
* **Primary Source**: [GitHub: lsils/benchmarks](https://github.com/lsils/benchmarks)
* **Contents**:
  * 23 natively combinational circuits distributed in 4 standard EDA formats (`.v`, `.vhd`, `.blif`, `.aig`):
    * **`arithmetic/`**: 10 circuits (`adder`, `bar`, `div`, `hyp`, `log2`, `max`, `multiplier`, `sin`, `sqrt`, `square`).
    * **`random_control/`**: 10 circuits (`arbiter`, `cavlc`, `ctrl`, `dec`, `i2c`, `int2float`, `mem_ctrl`, `priority`, `router`, `voter`).
  * **`best_results/`**: Tracks state-of-the-art LUT-6 logic optimization results (size and depth metrics).
