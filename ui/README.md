# Heart Disease Risk Predictor - Streamlit UI

## 🎯 Overview

A comprehensive web application for heart disease risk prediction using machine learning. This Streamlit app provides an intuitive interface for users to input their health data and receive real-time predictions with interactive visualizations.

## ✨ Features

### 🏥 **Prediction Page**
- **Interactive Input Form**: User-friendly form with sliders, dropdowns, and number inputs
- **Real-time Predictions**: Instant heart disease risk assessment
- **Risk Level Classification**: Low, Medium, or High risk with color-coded indicators
- **Probability Gauge**: Visual representation of prediction confidence
- **Medical Recommendations**: Personalized health advice based on risk level

### 📊 **Data Insights Page**
- **Age Distribution**: Histogram showing age patterns by heart disease status
- **Gender Analysis**: Bar charts for gender-based heart disease distribution
- **Chest Pain Analysis**: Distribution of chest pain types and outcomes
- **Correlation Heatmap**: Interactive feature correlation visualization

### ℹ️ **About Page**
- **Model Information**: Technical details about the ML model
- **Performance Metrics**: Accuracy, precision, recall, and F1-score
- **Medical Disclaimer**: Important health information
- **Data Privacy**: Information about data handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r ui/requirements.txt
   ```

2. **Ensure Model Files Exist**:
   - `models/heart_disease_pipeline.pkl`
   - `models/model_metadata.json`
   - `data/heart_disease_selected.csv`

3. **Run the Application**:
   ```bash
   # Option 1: Using the launcher script
   python run_streamlit.py
   
   # Option 2: Direct streamlit command
   cd ui && streamlit run app.py
   ```

4. **Access the App**:
   - Open your browser and go to: `http://localhost:8501`

## 📋 Input Parameters

The app accepts the following health parameters:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| Age | Slider | 20-100 | Patient's age in years |
| Sex | Dropdown | Female/Male | Biological sex |
| Chest Pain Type | Dropdown | 4 options | Type of chest pain experienced |
| Resting BP | Number | 90-200 mmHg | Systolic blood pressure |
| Cholesterol | Number | 100-600 mg/dl | Serum cholesterol level |
| Fasting Blood Sugar | Dropdown | Yes/No | Blood sugar > 120 mg/dl |
| Resting ECG | Dropdown | 3 options | Electrocardiogram results |
| Max Heart Rate | Slider | 60-202 | Maximum heart rate during exercise |
| Exercise Angina | Dropdown | Yes/No | Exercise-induced chest pain |
| ST Depression | Number | 0.0-6.2 | ST depression from exercise |
| ST Slope | Dropdown | 3 options | ST segment slope |

## 🎨 UI Features

### **Responsive Design**
- Wide layout optimized for desktop and tablet use
- Sidebar navigation for easy page switching
- Mobile-friendly input controls

### **Interactive Elements**
- Real-time form validation
- Dynamic risk level indicators
- Interactive Plotly visualizations
- Color-coded prediction results

### **User Experience**
- Clear medical terminology with helpful tooltips
- Intuitive navigation between pages
- Professional medical application styling
- Loading indicators for predictions

## 🔧 Technical Details

### **Model Integration**
- Loads trained Random Forest pipeline
- Handles feature preprocessing automatically
- Maintains feature order consistency
- Provides confidence scores

### **Data Visualization**
- Plotly-based interactive charts
- Real-time data loading
- Responsive chart sizing
- Professional color schemes

### **Performance**
- Cached model loading for fast startup
- Efficient prediction pipeline
- Optimized data processing
- Minimal memory footprint

## 🛡️ Safety & Privacy

### **Medical Disclaimer**
- Educational purposes only
- Not a substitute for professional medical advice
- Always consult healthcare professionals
- Results should not be used for self-diagnosis

### **Data Privacy**
- No data storage or transmission
- Local processing only
- No personal information collected
- Secure model inference

## 🐛 Troubleshooting

### **Common Issues**

1. **Model Not Found**:
   ```bash
   # Run the model export script first
   python model_export_deployment.py
   ```

2. **Missing Dependencies**:
   ```bash
   # Install all required packages
   pip install -r ui/requirements.txt
   ```

3. **Port Already in Use**:
   ```bash
   # Use a different port
   streamlit run app.py --server.port 8502
   ```

4. **Browser Not Opening**:
   - Manually navigate to `http://localhost:8501`
   - Check firewall settings

### **Performance Issues**
- Ensure sufficient RAM (2GB+ recommended)
- Close other applications if needed
- Use a modern web browser

## 📈 Model Performance

The underlying ML model achieves:
- **Accuracy**: 92.02%
- **Precision**: 92.80%
- **Recall**: 92.06%
- **F1-Score**: 92.43%
- **AUC**: 97.19%

## 🔄 Updates & Maintenance

### **Adding New Features**
1. Modify `app.py` in the ui directory
2. Test with `python test_streamlit.py`
3. Update requirements.txt if needed
4. Document changes in this README

### **Model Updates**
1. Retrain model using notebooks
2. Export new model with `model_export_deployment.py`
3. Restart Streamlit application
4. Verify predictions work correctly

## 📞 Support

For technical issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are present
4. Test with the provided test script

## 📄 License

This application is for educational and research purposes. Please ensure compliance with local regulations regarding medical software and data privacy.

---

**Built with ❤️ using Streamlit, Plotly, and Scikit-learn** 