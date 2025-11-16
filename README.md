# Liquid-based Nitrous Oxide Quantification
The respository provides a practical tool wrapped as a Python package for calculating nitrous oxide (N<sub>2</sub>O) emissions using a liquid N<sub>2</sub>O sensor. It details the workflow from data collection through robust preprocessing (IQR outlier removal, interpolation), to the estimation of the N<sub>2</sub>O mass-transfer coefficient (k<sub>L</sub>a<sub>N<sub>2</sub>O</sub>), calculation of the N<sub>2</sub>O flux, and visualization of results.

---

## 📦 Features

- **Modular class-based API** via `LiquidQuantifier`
- **Automated preprocessing**: outlier removal, interpolation, and data smoothing  
- **k<sub>L</sub>a estimation** using time-series oxygen and N<sub>2</sub>O sensor data  
- **Interactive visualizations** for comparing pre- and post-treatment data  
- **Excel data input/output** with automated parsing  

---

## ⚙️ Installation
* Ensure you have Python 3.8+ and the following dependencies:
pandas
numpy
matplotlib
openpyxl

Clone this repository and install in editable (development) mode:

```bash
git clone https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification.git
cd Liquid-based-Nitrous-Oxide-Quantification
# Install the package in editable mode
python -m pip install -e .

```

## 🚀 Quick start

* Option 1: In Anaconda Powershell Prompt
```bash
# Activate your interactive Python shell
python
# load the module
from liquidbased_quantification import LiquidQuantifier
# Initialize model with your Excel data
model = LiquidQuantifier("data/Monitoring_Data_Example.xlsx")
# Run full analysis pipeline
results = model.run_pipeline(show_plots=True)
```

* Option 2: In any Python interpreter under your Conda environment with the cloned package installed

(1) Open ['demo.py'](https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification-/blob/main/demo.py) and click Run (e.g., in VS Code or Spyder)
, OR,
(2) Open ['demo.ipynb'](https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification-/blob/main/demo.ipynb) in Jupyter Notebook and run all cells


---
## 🧩 Example Data

A placeholder Excel file is included under data/Monitoring_Data_Example.xlsx.
Replace this file with your actual monitoring dataset.
Your dataset should contain timestamped measurements of dissolved oxygen, N₂O, and related process variables.
Your dataset should keep the same header and unit as the provided template.

---

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

---

## 👩‍🔬 Citation

If you use this package in academic work, please cite as:
Wang, S., Duan, H. (2025). Chapter 3. Liquid based quantification methodology. The University of Queensland.

---

## 🤝 Contributing
Pull requests are welcome!
For major changes, please open an issue to discuss your proposed ideas first.

You may also contact the author directly using the information below.
Shuting Wang
School of Chemical Engineering,
The University of Queensland, Australia
📧 shuting.wang@uq.edu.au

