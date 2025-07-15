# 🚀 Heart Disease Prediction App - Deployment Guide

## 🌟 **Option 1: Streamlit Cloud (Recommended - FREE)**

### **Step 1: Prepare Your Code**
1. Make sure all files are in the correct structure:
   ```
   Heart_Disease_Project/
   ├── ui/
   │   ├── app.py
   │   ├── requirements.txt
   │   ├── .streamlit/
   │   │   └── config.toml
   │   └── models/
   │       ├── best_model.pkl
   │       └── model_metadata.json
   └── data/
       └── heart_disease_selected.csv
   ```

### **Step 2: Upload to GitHub**
1. Create a GitHub account if you don't have one
2. Create a new repository called `heart-disease-predictor`
3. Upload your `Heart_Disease_Project` folder to GitHub
4. Make sure the repository is public

### **Step 3: Deploy on Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `heart-disease-predictor`
5. Set the main file path: `Heart_Disease_Project/ui/app.py`
6. Click "Deploy"

### **Step 4: Share with Friends**
- Your app will be available at: `https://your-app-name.streamlit.app`
- Share this URL with your friends!

---

## 🌐 **Option 2: Heroku (Alternative)**

### **Step 1: Create Heroku Files**
Create these files in your project root:

**setup.sh:**
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > ~/.streamlit/config.toml
```

**Procfile:**
```
web: sh setup.sh && streamlit run Heart_Disease_Project/ui/app.py
```

### **Step 2: Deploy to Heroku**
1. Install Heroku CLI
2. Run these commands:
   ```bash
   heroku login
   heroku create your-app-name
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

---

## 🐳 **Option 3: Docker (Advanced)**

### **Step 1: Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Heart_Disease_Project/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Step 2: Build and Run**
```bash
docker build -t heart-disease-app .
docker run -p 8501:8501 heart-disease-app
```

---

## 📱 **Option 4: Local Network Sharing**

### **For Friends on Same Network:**
1. Find your computer's IP address:
   - Windows: `ipconfig` in CMD
   - Mac/Linux: `ifconfig` in Terminal

2. Run the app:
   ```bash
   cd Heart_Disease_Project/ui
   streamlit run app.py --server.address=0.0.0.0 --server.port=8501
   ```

3. Share the URL: `http://YOUR_IP_ADDRESS:8501`

---

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **Model not found**: Make sure model files are in the correct path
2. **Dependencies missing**: Check requirements.txt is complete
3. **Port issues**: Try different ports (8502, 8503, etc.)

### **File Path Fixes:**
If you get path errors, update the paths in `app.py`:
```python
# Change from:
model_path = "../models/best_model.pkl"

# To:
model_path = "models/best_model.pkl"
```

---

## 🎯 **Recommended: Streamlit Cloud**

**Why Streamlit Cloud is best:**
- ✅ **FREE** forever
- ✅ **Easy setup** (5 minutes)
- ✅ **Automatic updates** when you push to GitHub
- ✅ **No server management**
- ✅ **Professional URL**
- ✅ **Works on all devices**

**Your friends can access it from:**
- 📱 Mobile phones
- 💻 Laptops
- 🖥️ Desktop computers
- Any web browser

---

## 📞 **Need Help?**

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are uploaded to GitHub
3. Make sure requirements.txt is correct
4. Test locally first: `streamlit run app.py`

**Happy Deploying! 🚀** 