# Anomaly Detection Streamlit Dashboard

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/<ORG_OR_USER>/anomaly-detection-streamlit-binder/main?urlpath=proxy/8501/)

## 📌 Project Overview
This project provides an interactive **Anomaly Detection Dashboard** built with **Streamlit**. It allows you to:
- Train an anomaly detection model (Isolation Forest) on uploaded data
- Detect anomalies on evaluation datasets using ML and statistical rules
- Visualize anomalies with interactive charts
- Download results and persist models

## 🚀 Launch on Binder
Click the badge above to launch the app in your browser via [MyBinder](https://mybinder.org). No installation required!

## ✅ Features
- **Upload Datasets**: Training and evaluation CSV files
- **Feature Engineering**: Moving average, EWMA, z-score, volume variance
- **Model Training**: Isolation Forest with configurable contamination
- **Anomaly Detection**: ML-based + statistical thresholds
- **Visualization**: Interactive Plotly charts highlighting anomalies
- **Downloads**: Annotated CSV and trained model (.pkl)

## 🛠 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/<ORG_OR_USER>/anomaly-detection-streamlit-binder.git
cd anomaly-detection-streamlit-binder

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📂 Repository Structure
```
.
├── app.py                # Streamlit dashboard code
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version pin
└── .binder/              # Binder configuration
    ├── start             # Launch Streamlit + JupyterLab
    └── postBuild         # Enable proxy extension
```

## 🔍 Usage Instructions
1. Upload your **training dataset** (CSV) and **evaluation dataset**.
2. Configure parameters (rolling window, EWMA alpha, thresholds).
3. Train the model and run detection.
4. Explore anomalies in the interactive chart.
5. Download results and/or trained model.

## 📸 Screenshots
*(Add screenshots of the dashboard here)*

## 📜 License
MIT License

