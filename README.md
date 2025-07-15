

# ❤️ Heart Disease Prediction App

A modern, interactive web application for heart disease risk prediction, powered by machine learning and a beautiful, responsive UI. Built with Streamlit, Plotly, and Scikit-learn.

---

## 🚀 Features

- **Modern Hero Section**: Animated gradient, glassmorphism, and key stats
- **Interactive Input Form**: User-friendly, emoji-enhanced, with tooltips
- **Real-Time AI Predictions**: Instant risk assessment and confidence gauge
- **Personalized Recommendations**: Actionable advice based on your results
- **Advanced Visualizations**: Gauge, radar, and comparison charts
- **Professional Design**: Responsive, mobile-friendly, and visually stunning
- **Data Privacy**: All processing is local, no data is stored or transmitted

---

## 🖥️ Demo

https://abdalrhmanfadel25-ai-model-heart-disease-predictor-uiapp-kyxaqx.streamlit.app/ <!-- (Add a screenshot if available) -->

---

## 📦 Project Structure

```
Heart_Disease_Project/
│
├── ui/
│   ├── app.py                # Main Streamlit app
│   ├── requirements.txt      # UI dependencies
│   ├── assets/
│   │   └── style.css         # Custom styles
│   └── README.md             # UI-specific documentation
│
├── models/
│   ├── heart_disease_pipeline.pkl
│   ├── model_metadata.json
│   └── ...                   # Other model files
│
├── data/
│   └── heart_disease_selected.csv
│
├── run_streamlit.py          # Launcher script
├── model_export_deployment.py# Model export utility
├── test_streamlit.py         # Automated test script
└── README.md                 # (This file)
```

---

## ⚡ Quick Start

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd Heart_Disease_Project
```

### 2. **Install Dependencies**
```bash
pip install -r ui/requirements.txt
```

### 3. **Ensure Model & Data Files Exist**
- `models/heart_disease_pipeline.pkl`
- `models/model_metadata.json`
- `data/heart_disease_selected.csv`

If missing, run:
```bash
python model_export_deployment.py
```

### 4. **Run the App**
```bash
# Option 1: Using the launcher
python run_streamlit.py

# Option 2: Direct Streamlit command
cd ui
streamlit run app.py
```

### 5. **Open in Browser**
Go to: [http://localhost:8501](http://localhost:8501)

---

## 📝 Input Parameters

| Parameter            | Type      | Range/Options         | Description                        |
|----------------------|-----------|----------------------|------------------------------------|
| Age                  | Slider    | 18-100               | Patient's age in years             |
| Sex                  | Dropdown  | Female/Male          | Biological sex                     |
| Chest Pain Type      | Dropdown  | 4 options            | Type of chest pain                 |
| Resting BP           | Number    | 90-200 mmHg          | Systolic blood pressure            |
| Cholesterol          | Number    | 100-600 mg/dl        | Serum cholesterol level            |
| Fasting Blood Sugar  | Dropdown  | Yes/No               | >120 mg/dl                         |
| Resting ECG          | Dropdown  | 3 options            | Electrocardiogram results          |
| Max Heart Rate       | Slider    | 60-202               | During exercise                    |
| Exercise Angina      | Dropdown  | Yes/No               | Exercise-induced chest pain        |
| ST Depression        | Number    | 0.0-6.2              | ST depression from exercise        |
| ST Slope             | Dropdown  | 3 options            | ST segment slope                   |

---

## 🎨 UI & UX Highlights

- **Animated Gradient Hero Section** (scoped to hero only)
- **Glassmorphism Cards** with soft shadows
- **Emoji-enhanced Inputs** and tooltips
- **Gauge, Radar, and Comparison Charts** (Plotly)
- **Mobile-Responsive** and touch-friendly
- **Custom Google Fonts** (Poppins)
- **Professional color palette and transitions**

---

## 📊 Model & Performance

- **Model**: Random Forest Classifier (with StandardScaler)
- **Accuracy**: ~92%
- **Precision**: ~92%
- **Recall**: ~91%
- **AUC**: ~97%
- **Pipeline**: Preprocessing + Model in one `.pkl` file

---

## 🛡️ Safety & Privacy

- **Medical Disclaimer**: For educational purposes only. Not a substitute for professional medical advice.
- **Data Privacy**: No data is stored or sent. All processing is local.

---

## 🐳 Deployment

### **Streamlit Cloud (Recommended)**
- Push your repo to GitHub
- Deploy at [share.streamlit.io](https://share.streamlit.io)
- Set main file: `Heart_Disease_Project/ui/app.py`

### **Heroku**
- Add `Procfile` and `setup.sh` (see `README_DEPLOYMENT.md`)
- Deploy with Heroku CLI

### **Docker**
- Use the provided Dockerfile (see `README_DEPLOYMENT.md`)

---

## 🧪 Testing

Run the automated test script:
```bash
python test_streamlit.py
```
- Checks model loading, prediction, and input simulation

---

## 🛠️ Maintenance & Updates

- **To update the model**: Retrain, export with `model_export_deployment.py`, and restart the app
- **To add features**: Edit `ui/app.py`, test, and update documentation

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue or submit a PR.

---

## 📄 License

This project is for educational and research purposes. Please comply with local regulations regarding medical software and data privacy.

---

**Built with ❤️ using Streamlit, Plotly, and Scikit-learn**  
*Modern UI by Fadel, 2025*

---

Let me know if you want this written to your `README.md` file or need a version with a specific logo, screenshot, or additional section!
