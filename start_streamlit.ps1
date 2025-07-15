# Heart Disease Streamlit App Launcher
# PowerShell script to start the Streamlit application

Write-Host "🚀 Starting Heart Disease Prediction App..." -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "ui\app.py")) {
    Write-Host "❌ Error: ui\app.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the Heart_Disease_Project directory." -ForegroundColor Yellow
    exit 1
}

# Check if model files exist
if (-not (Test-Path "models\heart_disease_pipeline.pkl")) {
    Write-Host "❌ Error: Model file not found!" -ForegroundColor Red
    Write-Host "Please run the model export script first:" -ForegroundColor Yellow
    Write-Host "python model_export_deployment.py" -ForegroundColor White
    exit 1
}

Write-Host "✅ All files found!" -ForegroundColor Green
Write-Host "🌐 Starting Streamlit server..." -ForegroundColor Cyan
Write-Host "📱 The app will open in your browser at: http://localhost:8501" -ForegroundColor Yellow
Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan

try {
    # Change to ui directory and run streamlit
    Set-Location "ui"
    python -m streamlit run app.py --server.port 8501
}
catch {
    Write-Host "❌ Error starting Streamlit: $_" -ForegroundColor Red
}
finally {
    Write-Host "🛑 Streamlit server stopped." -ForegroundColor Yellow
} 