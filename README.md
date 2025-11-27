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

---
## 🧩 Example Data

* A placeholder Excel file is included under ['data/Monitoring_Data_Example.xlsx'](https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification/blob/21d9d6b2bafa2d38a378dd28d77513d84e06478f/data/Monitoring_Data_Example.xlsx).
Replace this file with your actual monitoring dataset.  
* Your dataset should contain timestamped measurements of dissolved N<sub>2</sub>O concentration (mgN<sub>2</sub>O-N/L), liquid temperature (°C) and pass airflow rate (m<sup>3</sup>/s). Please ensure that the units of all variables are converted to the specified units in advance, as this is required for the subsequent calculations.   
* Your dataset should keep the same header as the provided template.
* Please update the reactor depth (D<sub>R</sub>) and aeration field size (AerationFieldSize) in the demo scripts (i.e. ['demo.py'](https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification-/blob/main/demo.py) or ['demo.ipynb'](https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification-/blob/main/demo.ipynb) to match your plant dimensions.

---


## 🚀 Quick start

* Option 1: In Anaconda Powershell Prompt

(1) Directly run the ['demo.py'](https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification-/blob/main/demo.py)
```bash
python .\demo.py
```
(2) OR, call functions of this package in the interactive python, if you want to know a little bit more of the calculation process
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
(2) OR, Open ['demo.ipynb'](https://github.com/Julylle/Liquid-based-Nitrous-Oxide-Quantification-/blob/main/demo.ipynb) in Jupyter Notebook and run all cells



## 📜 License
This project is licensed under the MIT License, see the LICENSE file for details.

---

## 👩‍🔬 Citation

If you use this package in academic work, please cite as:
Wang, S., Duan, H. (2025). Chapter 3. Liquid based quantification methodology. The University of Queensland.

---

## 🤝 Contributing
Pull requests are welcome!
For major changes, please open an issue to discuss your proposed ideas first.

You may also contact the author directly using the information below:  

Shuting Wang  
School of Chemical Engineering,  
The University of Queensland, Australia  
📧 shuting.wang@uq.edu.au

