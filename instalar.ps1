# Script de Instalación Rápida para Windows
# Ejecuta este script después de instalar Python, CUDA y cuDNN

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  INSTALACIÓN AUTOMÁTICA" -ForegroundColor Cyan
Write-Host "  Detección de Números de Dorsal" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python
Write-Host "[1/6] Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ $pythonVersion encontrado" -ForegroundColor Green
} else {
    Write-Host "✗ Python no encontrado. Instálalo desde https://www.python.org" -ForegroundColor Red
    exit 1
}

# Verificar NVIDIA GPU
Write-Host ""
Write-Host "[2/6] Verificando GPU NVIDIA..." -ForegroundColor Yellow
$nvidiaSmi = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ GPU NVIDIA detectada" -ForegroundColor Green
} else {
    Write-Host "⚠ nvidia-smi no encontrado. Instala drivers NVIDIA" -ForegroundColor Yellow
}

# Crear entorno virtual
Write-Host ""
Write-Host "[3/6] Creando entorno virtual..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠ Entorno virtual ya existe, omitiendo..." -ForegroundColor Yellow
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Entorno virtual creado" -ForegroundColor Green
    } else {
        Write-Host "✗ Error al crear entorno virtual" -ForegroundColor Red
        exit 1
    }
}

# Activar entorno virtual
Write-Host ""
Write-Host "[4/6] Activando entorno virtual..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Actualizar pip
Write-Host ""
Write-Host "[5/6] Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Instalar PyTorch con CUDA
Write-Host ""
Write-Host "[6/6] Instalando dependencias..." -ForegroundColor Yellow
Write-Host "Esto puede tomar varios minutos..." -ForegroundColor Cyan

Write-Host ""
Write-Host "  -> Instalando PyTorch con soporte CUDA..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Error al instalar PyTorch" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  -> Instalando OpenCV..." -ForegroundColor Cyan
pip install opencv-python opencv-contrib-python

Write-Host ""
Write-Host "  -> Instalando paquetes científicos..." -ForegroundColor Cyan
pip install numpy pandas scipy h5py matplotlib imgaug

Write-Host ""
Write-Host "  -> Instalando Jupyter..." -ForegroundColor Cyan
pip install jupyter notebook ipython ipykernel

Write-Host ""
Write-Host "  -> Instalando utilidades..." -ForegroundColor Cyan
pip install tqdm Pillow

# Verificar instalación
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VERIFICANDO INSTALACIÓN" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Ejecutando script de verificación..." -ForegroundColor Yellow
python verificar_instalacion.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  ¡INSTALACIÓN COMPLETADA!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Próximos pasos:" -ForegroundColor Cyan
    Write-Host "  1. El entorno virtual ya está activado" -ForegroundColor White
    Write-Host "  2. Navega a los notebooks:" -ForegroundColor White
    Write-Host "     cd notebooks+utils+data" -ForegroundColor Yellow
    Write-Host "  3. Inicia Jupyter Notebook:" -ForegroundColor White
    Write-Host "     jupyter notebook" -ForegroundColor Yellow
    Write-Host "  4. Abre el notebook de demo:" -ForegroundColor White
    Write-Host "     05 - Bib Detection Validation & Demo.ipynb" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "  ADVERTENCIA" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "La instalación completó pero hubo algunas advertencias." -ForegroundColor Yellow
    Write-Host "Consulta el archivo MANUAL_INSTALACION.md para más detalles." -ForegroundColor Yellow
    Write-Host ""
}
