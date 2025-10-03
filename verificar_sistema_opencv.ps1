# ===================================================================
# ASISTENTE DE COMPILACION OPENCV + CUDA
# Script de verificacion y guia paso a paso
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  ASISTENTE: COMPILAR OPENCV CON CUDA" -ForegroundColor Cyan
Write-Host "  Para usar GPU con deteccion YOLO" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Este script verifica tu sistema y te guia en el proceso" -ForegroundColor Yellow
Write-Host "Tiempo estimado: 2-3 horas" -ForegroundColor Yellow
Write-Host ""

$pasoActual = 1

# ===================================================================
# PASO 1: VERIFICAR REQUISITOS
# ===================================================================

Write-Host "[$pasoActual/10] VERIFICANDO REQUISITOS..." -ForegroundColor Green
Write-Host ""

# GPU NVIDIA
Write-Host "[GPU] Verificando GPU NVIDIA..." -ForegroundColor Yellow
$nvidiaGpu = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    $gpuName = $nvidiaGpu | Select-String "GeForce" | Select-Object -First 1
    Write-Host "   OK: $gpuName" -ForegroundColor Green
} else {
    Write-Host "   [X] GPU NVIDIA no detectada" -ForegroundColor Red
    Write-Host "       Instala drivers desde: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    exit 1
}

# CUDA Version
Write-Host ""
Write-Host "[CUDA] Verificando CUDA Toolkit..." -ForegroundColor Yellow
$cudaVersion = $nvidiaGpu | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
if ($cudaVersion) {
    Write-Host "   OK: CUDA $cudaVersion detectado" -ForegroundColor Green
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion"
    
    if (Test-Path $cudaPath) {
        Write-Host "   OK: CUDA Toolkit en: $cudaPath" -ForegroundColor Green
    } else {
        Write-Host "   [!] CUDA Runtime detectado pero Toolkit no encontrado" -ForegroundColor Yellow
        Write-Host "       Descarga CUDA Toolkit desde:" -ForegroundColor Yellow
        Write-Host "       https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    }
} else {
    Write-Host "   [X] CUDA no detectado" -ForegroundColor Red
    exit 1
}

# Python
Write-Host ""
Write-Host "[PYTHON] Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   OK: $pythonVersion" -ForegroundColor Green
    
    # Ruta de Python
    $pythonPath = python -c "import sys; print(sys.executable)" 2>&1
    Write-Host "   Ubicacion: $pythonPath" -ForegroundColor Gray
} else {
    Write-Host "   [X] Python no encontrado" -ForegroundColor Red
    Write-Host "       Descarga desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Espacio en disco
Write-Host ""
Write-Host "[DISCO] Verificando espacio disponible..." -ForegroundColor Yellow
$drive = Get-PSDrive C
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "   Espacio libre en C:\: $freeSpaceGB GB" -ForegroundColor Gray
if ($freeSpaceGB -lt 20) {
    Write-Host "   [!] ADVERTENCIA: Se recomienda al menos 20 GB libres" -ForegroundColor Yellow
    Write-Host "       Tienes: $freeSpaceGB GB" -ForegroundColor Yellow
} else {
    Write-Host "   OK: Espacio suficiente" -ForegroundColor Green
}

$pasoActual++

# ===================================================================
# PASO 2: VERIFICAR HERRAMIENTAS
# ===================================================================

Write-Host ""
Write-Host "[$pasoActual/10] VERIFICANDO HERRAMIENTAS DE DESARROLLO..." -ForegroundColor Green
Write-Host ""

# Visual Studio
Write-Host "[VS] Verificando Visual Studio..." -ForegroundColor Yellow
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
if (Test-Path $vsPath) {
    Write-Host "   OK: Visual Studio 2022 encontrado" -ForegroundColor Green
    Write-Host "   Ubicacion: $vsPath" -ForegroundColor Gray
} else {
    Write-Host "   [X] Visual Studio 2022 no encontrado" -ForegroundColor Red
    Write-Host "       ACCION REQUERIDA:" -ForegroundColor Yellow
    Write-Host "       1. Descarga Visual Studio 2022 Community:" -ForegroundColor White
    Write-Host "          https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
    Write-Host "       2. Instala con workload: Desktop development with C++" -ForegroundColor White
    Write-Host ""
    $continuar = Read-Host "       Ya instalaste Visual Studio? (S/N)"
    if ($continuar -ne "S" -and $continuar -ne "s") {
        Write-Host ""
        Write-Host "       Instala Visual Studio y ejecuta este script de nuevo" -ForegroundColor Yellow
        Read-Host "       Presiona Enter para salir"
        exit 1
    }
}

# CMake
Write-Host ""
Write-Host "[CMAKE] Verificando CMake..." -ForegroundColor Yellow
$cmakeVersion = cmake --version 2>&1
if ($LASTEXITCODE -eq 0) {
    $version = $cmakeVersion | Select-String "version (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
    Write-Host "   OK: CMake $version" -ForegroundColor Green
} else {
    Write-Host "   [X] CMake no encontrado" -ForegroundColor Red
    Write-Host "       ACCION REQUERIDA:" -ForegroundColor Yellow
    Write-Host "       1. Descarga CMake:" -ForegroundColor White
    Write-Host "          https://cmake.org/download/" -ForegroundColor Cyan
    Write-Host "       2. Durante instalacion, marca: Add CMake to PATH" -ForegroundColor White
    Write-Host ""
    $continuar = Read-Host "       Ya instalaste CMake? (S/N)"
    if ($continuar -ne "S" -and $continuar -ne "s") {
        Write-Host ""
        Write-Host "       Instala CMake y ejecuta este script de nuevo" -ForegroundColor Yellow
        Read-Host "       Presiona Enter para salir"
        exit 1
    }
}

$pasoActual++

# ===================================================================
# PASO 3: VERIFICAR cuDNN
# ===================================================================

Write-Host ""
Write-Host "[$pasoActual/10] VERIFICANDO cuDNN..." -ForegroundColor Green
Write-Host ""

$cudnnDll = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion\bin\cudnn64_8.dll"
if (Test-Path $cudnnDll) {
    Write-Host "   OK: cuDNN instalado" -ForegroundColor Green
    Write-Host "   Ubicacion: $cudnnDll" -ForegroundColor Gray
} else {
    Write-Host "   [X] cuDNN no encontrado" -ForegroundColor Red
    Write-Host "       ACCION REQUERIDA:" -ForegroundColor Yellow
    Write-Host "       1. Descarga cuDNN (requiere cuenta NVIDIA gratuita):" -ForegroundColor White
    Write-Host "          https://developer.nvidia.com/cudnn" -ForegroundColor Cyan
    Write-Host "       2. Selecciona version para CUDA $cudaVersion" -ForegroundColor White
    Write-Host "       3. Extrae archivos y copia a carpeta CUDA:" -ForegroundColor White
    Write-Host "          bin/*.dll    -> C:\Program Files\...\CUDA\v$cudaVersion\bin\" -ForegroundColor White
    Write-Host "          include/*.h  -> C:\Program Files\...\CUDA\v$cudaVersion\include\" -ForegroundColor White
    Write-Host "          lib/*.lib    -> C:\Program Files\...\CUDA\v$cudaVersion\lib\x64\" -ForegroundColor White
    Write-Host ""
    Write-Host "       Lee la guia completa en: GUIA_COMPILAR_OPENCV_CUDA.txt" -ForegroundColor Cyan
    Write-Host "       Seccion 4: CONFIGURAR CUDA Y cuDNN" -ForegroundColor Cyan
    Write-Host ""
    $continuar = Read-Host "       Ya instalaste cuDNN? (S/N)"
    if ($continuar -ne "S" -and $continuar -ne "s") {
        Write-Host ""
        Write-Host "       Instala cuDNN y ejecuta este script de nuevo" -ForegroundColor Yellow
        Read-Host "       Presiona Enter para salir"
        exit 1
    }
}

$pasoActual++

# ===================================================================
# RESUMEN Y SIGUIENTES PASOS
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  VERIFICACION COMPLETADA" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "ESTADO DE TU SISTEMA:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  GPU:           $gpuName" -ForegroundColor White
Write-Host "  CUDA:          $cudaVersion" -ForegroundColor White
Write-Host "  Python:        $pythonVersion" -ForegroundColor White
Write-Host "  Visual Studio: " -NoNewline -ForegroundColor White
if (Test-Path $vsPath) { Write-Host "Instalado" -ForegroundColor Green } else { Write-Host "Falta" -ForegroundColor Red }
Write-Host "  CMake:         " -NoNewline -ForegroundColor White
if ($LASTEXITCODE -eq 0) { Write-Host "Instalado" -ForegroundColor Green } else { Write-Host "Falta" -ForegroundColor Red }
Write-Host "  cuDNN:         " -NoNewline -ForegroundColor White
if (Test-Path $cudnnDll) { Write-Host "Instalado" -ForegroundColor Green } else { Write-Host "Falta" -ForegroundColor Red }
Write-Host "  Espacio libre: $freeSpaceGB GB" -ForegroundColor White
Write-Host ""

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  SIGUIENTES PASOS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Ahora debes:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. DESCARGAR CODIGO FUENTE DE OPENCV" -ForegroundColor White
Write-Host "   - OpenCV 4.8.0:" -ForegroundColor Gray
Write-Host "     https://github.com/opencv/opencv/archive/4.8.0.zip" -ForegroundColor Cyan
Write-Host "   - OpenCV Contrib 4.8.0:" -ForegroundColor Gray
Write-Host "     https://github.com/opencv/opencv_contrib/archive/4.8.0.zip" -ForegroundColor Cyan
Write-Host ""

Write-Host "2. EXTRAER ARCHIVOS" -ForegroundColor White
Write-Host "   Crea estructura:" -ForegroundColor Gray
Write-Host "   C:\opencv_build\" -ForegroundColor Cyan
Write-Host "   ├── opencv\" -ForegroundColor Cyan
Write-Host "   ├── opencv_contrib\" -ForegroundColor Cyan
Write-Host "   └── build\" -ForegroundColor Cyan
Write-Host ""

Write-Host "3. CONFIGURAR CON CMAKE" -ForegroundColor White
Write-Host "   - Abre cmake-gui" -ForegroundColor Gray
Write-Host "   - Source: C:/opencv_build/opencv" -ForegroundColor Gray
Write-Host "   - Build:  C:/opencv_build/build" -ForegroundColor Gray
Write-Host "   - Configure + ajustar opciones + Generate" -ForegroundColor Gray
Write-Host ""

Write-Host "4. COMPILAR CON VISUAL STUDIO" -ForegroundColor White
Write-Host "   - Abrir proyecto en VS 2022" -ForegroundColor Gray
Write-Host "   - Cambiar a Release x64" -ForegroundColor Gray
Write-Host "   - Build Solution (1-2 horas)" -ForegroundColor Gray
Write-Host ""

Write-Host "5. INSTALAR Y CONFIGURAR" -ForegroundColor White
Write-Host "   - Build proyecto INSTALL" -ForegroundColor Gray
Write-Host "   - Copiar archivos Python" -ForegroundColor Gray
Write-Host "   - Agregar DLLs al PATH" -ForegroundColor Gray
Write-Host ""

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "LEE LA GUIA COMPLETA:" -ForegroundColor Yellow
Write-Host "  GUIA_COMPILAR_OPENCV_CUDA.txt" -ForegroundColor White
Write-Host ""
Write-Host "Tiene instrucciones detalladas para cada paso" -ForegroundColor Gray
Write-Host ""

Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "NOTA IMPORTANTE:" -ForegroundColor Yellow
Write-Host "Este proceso es complejo y toma 2-3 horas." -ForegroundColor White
Write-Host "Si prefieres una solucion mas rapida:" -ForegroundColor White
Write-Host ""
Write-Host "  OPCION A: Usar CPU optimizado (ya listo)" -ForegroundColor Cyan
Write-Host "    python mi_detector_rapido.py --modo camara" -ForegroundColor White
Write-Host "    Rendimiento: 20-30 FPS (suficiente)" -ForegroundColor Gray
Write-Host ""
Write-Host "  OPCION B: Compilar OpenCV (esta guia)" -ForegroundColor Cyan
Write-Host "    Tiempo: 2-3 horas" -ForegroundColor White
Write-Host "    Rendimiento: 60-100 FPS" -ForegroundColor Gray
Write-Host ""

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$decision = Read-Host "Continuar con compilacion? (S=Si, N=Usar CPU) [S/N]"

if ($decision -eq "N" -or $decision -eq "n") {
    Write-Host ""
    Write-Host "OK, usando version CPU optimizada" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ejecuta: python mi_detector_rapido.py --modo camara" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Excelente! Sigue la guia paso a paso:" -ForegroundColor Green
    Write-Host "  GUIA_COMPILAR_OPENCV_CUDA.txt" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Si tienes problemas, consulta la seccion 11:" -ForegroundColor Yellow
    Write-Host "  SOLUCION DE PROBLEMAS COMUNES" -ForegroundColor White
    Write-Host ""
}

Read-Host "Presiona Enter para finalizar"
