# Hardware Trojan Power & EM Side-Channel Dataset

- **Identifier / DOI**: [10.21227/9fwb-8978](https://dx.doi.org/10.21227/9fwb-8978)
- **Official Portal**: [IEEE DataPort](https://ieee-dataport.org/open-access/hardware-trojan-power-em-side-channel-dataset)
- **Authors**: Rozhin Yasaei, Sina Faezi, Mohammad Abdullah Al Faruque (University of California, Irvine)
- **Primary Publication**: *"Brain-Inspired Golden Chip Free Hardware Trojan Detection"*, IEEE Transactions on Information Forensics and Security (IEEE TIFS), 2021.
- **License**: Creative Commons Attribution (CC BY 4.0)

---

## Dataset Overview

This dataset contains measured physical power and electromagnetic (EM) side-channel execution traces recorded from FPGA implementations of Trust-Hub benchmark circuits (AES cryptographic cores) under controlled experimental conditions (Temperature: 25°C).

For each hardware Trojan variant, side-channel data is collected under two distinct operational scenarios:
1. **Trojan Inactive**: Captures baseline circuit power/EM consumption plus any static leakage footprints from the dormant Trojan logic.
2. **Trojan Activated**: Captures active dynamic power and EM variations caused by Trojan trigger activation and payload execution.

This dual-mode data allows benchmarking and training machine learning / deep learning models (CNNs, LSTMs, Siamese networks) for golden chip-free Hardware Trojan detection.

---

## Included Trace Archives

| Archive Name | Tested Circuit Variant | File Size | Description |
|---|---|---|---|
| `AES-T400_power_Temp25C.zip` | AES-T400 | 163.77 MB | Power side-channel traces at 25°C |
| `AES-T500_power_Temp25C.zip` | AES-T500 | 162.92 MB | Power side-channel traces at 25°C |
| `AES-T600_power_Temp25C.zip` | AES-T600 | 164.67 MB | Power side-channel traces at 25°C |
| `AES-T700_power_Temp25C.zip` | AES-T700 | 156.14 MB | Power side-channel traces at 25°C |
| `AES-T800_power_Temp25C.zip` | AES-T800 | 160.24 MB | Power side-channel traces at 25°C |
| `AES-T1000_power_Temp25C.zip` | AES-T1000 | 177.15 MB | Power side-channel traces at 25°C |
| `AES-T1100_power_Temp25C.zip` | AES-T1100 | 177.75 MB | Power side-channel traces at 25°C |
| `AES-T1300_power_Temp25C.zip` | AES-T1300 | 163.98 MB | Power side-channel traces at 25°C |
| `AES-T1400_power_Temp25C.zip` | AES-T1400 | 164.62 MB | Power side-channel traces at 25°C |
| `AES-T1600_power_Temp25C.zip` | AES-T1600 | 164.57 MB | Power side-channel traces at 25°C |
| `AES-T1800_power_Temp25C.zip` | AES-T1800 | 165.01 MB | Power side-channel traces at 25°C |
| `AES-T2000_power_Temp25C.zip` | AES-T2000 | 164.68 MB | Power side-channel traces at 25°C |

**Total Archive Size**: ~2.0 GB

---

## How to Download

IEEE DataPort provides open access to this dataset for any user with a **free IEEE web account** (an IEEE society membership is **not** required).

### Option 1: Browser Download (Recommended)
1. Navigate to: [https://ieee-dataport.org/open-access/hardware-trojan-power-em-side-channel-dataset](https://ieee-dataport.org/open-access/hardware-trojan-power-em-side-channel-dataset)
2. Click **Login** / **Create Free Account** and sign in.
3. Scroll to the **Dataset Files** section and click the download button for each `.zip` file.
4. Move the downloaded `.zip` files into `datasets/ieee_dataport_side_channel/data/`.

### Option 2: Automated Script with Session Cookie
Once logged into IEEE DataPort in your browser:
1. Open DevTools (F12) -> **Application** / **Storage** -> **Cookies** -> `https://ieee-dataport.org`.
2. Copy the cookie string (key starting with `SSESS...`).
3. Run:
```bash
python3 download_dataset.py --session-cookie "SSESS<your_cookie_name>=<your_cookie_val>"
```
