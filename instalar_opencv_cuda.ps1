# ===================================================================
# INSTALACION DE OPENCV CON SOPORTE CUDA
# Este script instala OpenCV compilado con CUDA para usar GPU
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  INSTALACION OPENCV CON CUDA" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que el entorno virtual este activo
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[!] ADVERTENCIA: Entorno virtual no detectado" -ForegroundColor Yellow
    Write-Host "    Activando entorno virtual..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

Write-Host "[INFO] Este proceso instalara OpenCV con soporte CUDA" -ForegroundColor Cyan
Write-Host "       Se desinstalar OpenCV actual y se instalara version con CUDA" -ForegroundColor Cyan
Write-Host ""

# Verificar version CUDA
Write-Host "[1/4] Verificando version CUDA..." -ForegroundColor Yellow
$cudaVersion = nvidia-smi 2>&1 | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }

if ($cudaVersion) {
    Write-Host "   OK: CUDA $cudaVersion detectado" -ForegroundColor Green
} else {
    Write-Host "   [!] No se pudo detectar version CUDA" -ForegroundColor Yellow
    Write-Host "   Continuando de todas formas..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[2/4] Desinstalando OpenCV actual..." -ForegroundColor Yellow
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>$null

Write-Host ""
Write-Host "[3/4] Instalando OpenCV con CUDA (opencv-contrib-python)..." -ForegroundColor Yellow
Write-Host "   [INFO] Buscando paquete precompilado..." -ForegroundColor Cyan
Write-Host ""

# OPCION A: Intentar con paquete no oficial de opencv-contrib-python con CUDA
Write-Host "   Intentando instalar desde repositorio no oficial..." -ForegroundColor Cyan
pip install opencv-contrib-python==4.8.0.74

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "   [!] Fallo la instalacion desde repositorio no oficial" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Probando con opencv-python normal (sin CUDA)..." -ForegroundColor Yellow
    pip install opencv-python opencv-contrib-python
    
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host "  ATENCION: OPENCV SIN CUDA INSTALADO" -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Para usar GPU con OpenCV necesitas:" -ForegroundColor White
    Write-Host ""
    Write-Host "OPCION 1 (Recomendada): Compilar OpenCV desde fuente" -ForegroundColor Cyan
    Write-Host "  - Descargar codigo fuente de OpenCV" -ForegroundColor White
    Write-Host "  - Compilar con CMake y Visual Studio" -ForegroundColor White
    Write-Host "  - Habilitar WITH_CUDA=ON" -ForegroundColor White
    Write-Host "  - Guia: https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html" -ForegroundColor White
    Write-Host ""
    Write-Host "OPCION 2: Usar CPU (funciona bien para inferencia)" -ForegroundColor Cyan
    Write-Host "  - El detector ya funciona con CPU" -ForegroundColor White
    Write-Host "  - Rendimiento: ~20-30 FPS en RTX 3050" -ForegroundColor White
    Write-Host ""
    Write-Host "OPCION 3: Usar PyTorch para deteccion (alternativa)" -ForegroundColor Cyan
    Write-Host "  - Modificar detector para usar PyTorch en lugar de OpenCV DNN" -ForegroundColor White
    Write-Host "  - PyTorch ya tiene CUDA funcionando" -ForegroundColor White
    Write-Host ""
    
    Read-Host "Presiona Enter para continuar"
    exit 1
}

Write-Host ""
Write-Host "[4/4] Verificando instalacion..." -ForegroundColor Yellow
Write-Host ""

$verificacion = python -c @"
import cv2
import sys

print(f'OpenCV version: {cv2.__version__}')
print(f'OpenCV build info:')

# Verificar CUDA
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f'  CUDA devices: {cuda_count}')
    if cuda_count > 0:
        print('  ✓ CUDA DISPONIBLE!')
        sys.exit(0)
    else:
        print('  ✗ CUDA no disponible (sin dispositivos)')
        sys.exit(1)
except Exception as e:
    print(f'  ✗ CUDA no disponible: {e}')
    sys.exit(1)
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  EXITO: OPENCV CON CUDA INSTALADO" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ahora puedes usar el detector con GPU:" -ForegroundColor Cyan
    Write-Host "  python mi_detector.py --modo camara" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host "  ERROR: OPENCV INSTALADO PERO SIN CUDA" -ForegroundColor Red
    Write-Host "================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "El paquete precompilado no tiene CUDA habilitado." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Opciones:" -ForegroundColor Cyan
    Write-Host "  1. Compilar OpenCV desde fuente con CUDA" -ForegroundColor White
    Write-Host "  2. Usar CPU (agrega --cpu al comando)" -ForegroundColor White
    Write-Host "  3. Modificar detector para usar PyTorch" -ForegroundColor White
    Write-Host ""
    
    Read-Host "Presiona Enter para salir"
    exit 1
}
