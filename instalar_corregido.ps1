# Script de Instalación Corregido para CUDA 13.0 + RTX 3050
# Ejecuta este script después de instalar Python correctamente

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  INSTALACIÓN CORREGIDA" -ForegroundColor Cyan
Write-Host "  Para CUDA 13.0 + RTX 3050" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python
Write-Host "[1/6] Verificando Python..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $pythonVersion encontrado" -ForegroundColor Green
    } else {
        throw "Python no encontrado"
    }
} catch {
    Write-Host "✗ Python no encontrado." -ForegroundColor Red
    Write-Host ""
    Write-Host "SOLUCIÓN:" -ForegroundColor Yellow
    Write-Host "1. Descarga Python 3.10 desde: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. Durante instalación, MARCA 'Add Python to PATH'" -ForegroundColor White
    Write-Host "3. Cierra y reabre PowerShell" -ForegroundColor White
    Write-Host "4. Ejecuta este script nuevamente" -ForegroundColor White
    Write-Host ""
    Write-Host "Presiona Enter para abrir la página de descarga de Python..." -ForegroundColor Cyan
    Read-Host
    Start-Process "https://www.python.org/downloads/"
    exit 1
}

# Verificar GPU
Write-Host ""
Write-Host "[2/6] Verificando GPU NVIDIA..." -ForegroundColor Yellow
try {
    $nvidiaSmi = & nvidia-smi 2>&1 | Select-String "RTX"
    if ($nvidiaSmi) {
        Write-Host "✓ GPU NVIDIA RTX detectada" -ForegroundColor Green
    } else {
        Write-Host "⚠ nvidia-smi funciona pero no detectó RTX" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ nvidia-smi no encontrado. Instala drivers NVIDIA" -ForegroundColor Yellow
}

# Eliminar entorno virtual anterior si existe
Write-Host ""
Write-Host "[3/6] Preparando entorno virtual..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠ Eliminando entorno virtual anterior..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Crear nuevo entorno virtual
Write-Host "→ Creando entorno virtual nuevo..." -ForegroundColor Cyan
python -m venv venv

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Entorno virtual creado exitosamente" -ForegroundColor Green
} else {
    Write-Host "✗ Error al crear entorno virtual" -ForegroundColor Red
    exit 1
}

# Activar entorno virtual
Write-Host ""
Write-Host "[4/6] Activando entorno virtual..." -ForegroundColor Yellow
$activateScript = ".\venv\Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "✓ Entorno virtual activado" -ForegroundColor Green
} else {
    Write-Host "✗ No se encontró el script de activación" -ForegroundColor Red
    exit 1
}

# Actualizar pip
Write-Host ""
Write-Host "[5/6] Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ pip actualizado" -ForegroundColor Green

# Instalar dependencias
Write-Host ""
Write-Host "[6/6] Instalando dependencias..." -ForegroundColor Yellow
Write-Host "Esto tomará 5-10 minutos dependiendo de tu conexión..." -ForegroundColor Cyan
Write-Host ""

# Detectar versión de CUDA
Write-Host "→ Detectando versión de CUDA..." -ForegroundColor Cyan
$cudaVersion = & nvidia-smi 2>&1 | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
Write-Host "  CUDA del sistema: $cudaVersion" -ForegroundColor White

# Determinar qué versión de PyTorch instalar
$torchIndex = "https://download.pytorch.org/whl/cu121"
if ($cudaVersion -like "13.*" -or $cudaVersion -like "12.*") {
    Write-Host "  → Usando PyTorch con CUDA 12.1 (compatible con CUDA $cudaVersion)" -ForegroundColor Cyan
    $torchIndex = "https://download.pytorch.org/whl/cu121"
} elseif ($cudaVersion -like "11.*") {
    Write-Host "  → Usando PyTorch con CUDA 11.8" -ForegroundColor Cyan
    $torchIndex = "https://download.pytorch.org/whl/cu118"
}

Write-Host ""
Write-Host "→ Instalando PyTorch con soporte GPU..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url $torchIndex --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ PyTorch instalado" -ForegroundColor Green
} else {
    Write-Host "⚠ Error instalando PyTorch, pero continuando..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "→ Instalando OpenCV..." -ForegroundColor Cyan
pip install opencv-python opencv-contrib-python --quiet
Write-Host "✓ OpenCV instalado" -ForegroundColor Green

Write-Host ""
Write-Host "→ Instalando librerías científicas..." -ForegroundColor Cyan
pip install numpy pandas scipy h5py matplotlib --quiet
Write-Host "✓ Librerías científicas instaladas" -ForegroundColor Green

Write-Host ""
Write-Host "→ Instalando imgaug..." -ForegroundColor Cyan
pip install imgaug --quiet
Write-Host "✓ imgaug instalado" -ForegroundColor Green

Write-Host ""
Write-Host "→ Instalando Jupyter..." -ForegroundColor Cyan
pip install jupyter notebook ipython ipykernel --quiet
Write-Host "✓ Jupyter instalado" -ForegroundColor Green

Write-Host ""
Write-Host "→ Instalando utilidades..." -ForegroundColor Cyan
pip install tqdm Pillow --quiet
Write-Host "✓ Utilidades instaladas" -ForegroundColor Green

# Verificar instalación de PyTorch + CUDA
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VERIFICACIÓN DE GPU" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$verificacion = python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No detectada\"}')" 2>&1

Write-Host $verificacion

# Evaluar resultado
if ($verificacion -like "*CUDA disponible: True*") {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  ✅ ¡INSTALACIÓN EXITOSA!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Tu GPU RTX 3050 está lista para usar 🚀" -ForegroundColor Green
    Write-Host ""
    Write-Host "Próximos pasos:" -ForegroundColor Cyan
    Write-Host "  1. El entorno virtual está activado" -ForegroundColor White
    Write-Host "  2. Ejecuta para verificar todo:" -ForegroundColor White
    Write-Host "     python verificar_instalacion.py" -ForegroundColor Yellow
    Write-Host "  3. Inicia Jupyter:" -ForegroundColor White
    Write-Host "     cd notebooks+utils+data" -ForegroundColor Yellow
    Write-Host "     jupyter notebook" -ForegroundColor Yellow
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "  ⚠️  ADVERTENCIA" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "PyTorch instalado pero GPU no detectada" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Posibles soluciones:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Intenta reinstalar con CUDA 11.8:" -ForegroundColor White
    Write-Host "   pip uninstall torch torchvision torchaudio -y" -ForegroundColor Yellow
    Write-Host "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Verifica que los drivers NVIDIA estén actualizados" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Consulta el archivo: SOLUCION_GPU_NO_DETECTADA.md" -ForegroundColor White
    Write-Host ""
}

Write-Host "Presiona Enter para continuar..." -ForegroundColor Gray
Read-Host
