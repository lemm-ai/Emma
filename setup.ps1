# EMMA Setup Script
# Run this script to complete the setup

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  EMMA Setup Script" -ForegroundColor Cyan
Write-Host "  Gamahea / LEMM Project" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonCmd = "D:/2025-vibe-coding/Emma/.venv/Scripts/python.exe"

if (Test-Path $pythonCmd) {
    $version = & $pythonCmd --version
    Write-Host "  $version" -ForegroundColor Green
} else {
    Write-Host "  Virtual environment not found!" -ForegroundColor Red
    Write-Host "  Please run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& $pythonCmd -m pip install --upgrade pip
& $pythonCmd -m pip install -r requirements.txt

Write-Host ""

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
$dirs = @("data", "logs", "output", "output/clips", "models")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}

Write-Host ""

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "  .env file created from .env.example" -ForegroundColor Green
    Write-Host "  Please edit .env to configure your settings" -ForegroundColor Yellow
} else {
    Write-Host ".env file already exists" -ForegroundColor Gray
}

Write-Host ""

# Display next steps
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Download AI model weights to models/ directory" -ForegroundColor White
Write-Host "   See README.md for model download links" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Configure settings in config.yaml and .env" -ForegroundColor White
Write-Host ""
Write-Host "3. Run EMMA:" -ForegroundColor White
Write-Host "   python app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Open browser to: http://localhost:7860" -ForegroundColor White
Write-Host ""
Write-Host "For help, see README.md or visit the repository" -ForegroundColor Gray
Write-Host ""
