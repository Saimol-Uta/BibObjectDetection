# ===================================================================
# INSTALAR OCR PARA LECTURA DE NÚMEROS EN DORSALES
# Instala EasyOCR para leer los números de los dorsales
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  INSTALACION: OCR PARA LECTURA DE DORSALES" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar entorno virtual
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[1/4] Activando entorno virtual..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error: No se pudo activar entorno virtual" -ForegroundColor Red
        Write-Host "    Ejecuta primero: .\instalar_corregido.ps1" -ForegroundColor Yellow
        Read-Host "Presiona Enter para salir"
        exit 1
    }
} else {
    Write-Host "[1/4] Entorno virtual activo" -ForegroundColor Green
}

Write-Host ""
Write-Host "OPCIONES DE OCR:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  A) EasyOCR (RECOMENDADO)" -ForegroundColor Cyan
Write-Host "     + Mayor precisión" -ForegroundColor Green
Write-Host "     + Funciona mejor con dorsales" -ForegroundColor Green
Write-Host "     - Descarga ~500MB de modelos" -ForegroundColor Yellow
Write-Host "     - Primera ejecución más lenta" -ForegroundColor Yellow
Write-Host ""
Write-Host "  B) Tesseract (ALTERNATIVA)" -ForegroundColor Cyan
Write-Host "     + Más rápido de instalar" -ForegroundColor Green
Write-Host "     + Menos espacio" -ForegroundColor Green
Write-Host "     - Menor precisión" -ForegroundColor Yellow
Write-Host "     - Requiere instalación adicional de Tesseract" -ForegroundColor Yellow
Write-Host ""

$opcion = Read-Host "Selecciona opción (A/B) [A por defecto]"

if ($opcion -eq "" -or $opcion -eq "A" -or $opcion -eq "a") {
    # Instalar EasyOCR
    Write-Host ""
    Write-Host "[2/4] Instalando dependencias de Excel..." -ForegroundColor Yellow
    pip install pandas openpyxl --quiet
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error al instalar pandas/openpyxl" -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    Write-Host "   OK: pandas y openpyxl instalados" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "[3/4] Instalando EasyOCR..." -ForegroundColor Yellow
    Write-Host "      (Esto puede tardar 5-10 minutos)" -ForegroundColor Gray
    pip install easyocr
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error al instalar EasyOCR" -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    Write-Host "   OK: EasyOCR instalado" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "[4/4] Verificando instalación..." -ForegroundColor Yellow
    python -c "import easyocr; import pandas; import openpyxl; print('Todas las dependencias instaladas')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host "  INSTALACION COMPLETADA - EasyOCR" -ForegroundColor Green
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "NOTA IMPORTANTE:" -ForegroundColor Yellow
        Write-Host "  La primera vez que ejecutes el detector, EasyOCR descargará" -ForegroundColor White
        Write-Host "  modelos adicionales (~500MB). Esto es normal y solo ocurre" -ForegroundColor White
        Write-Host "  una vez. Ten paciencia." -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host "[X] Error en verificación" -ForegroundColor Red
    }
    
} elseif ($opcion -eq "B" -or $opcion -eq "b") {
    # Instalar Tesseract
    Write-Host ""
    Write-Host "[2/4] Instalando dependencias de Excel..." -ForegroundColor Yellow
    pip install pandas openpyxl --quiet
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error al instalar pandas/openpyxl" -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    Write-Host "   OK: pandas y openpyxl instalados" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "[3/4] Instalando pytesseract..." -ForegroundColor Yellow
    pip install pytesseract --quiet
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error al instalar pytesseract" -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    Write-Host "   OK: pytesseract instalado" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "[4/4] Verificando instalación..." -ForegroundColor Yellow
    python -c "import pytesseract; import pandas; import openpyxl; print('Dependencias Python instaladas')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor Yellow
        Write-Host "  INSTALACION PYTHON COMPLETADA - Tesseract" -ForegroundColor Yellow
        Write-Host "================================================================" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "PASO ADICIONAL REQUERIDO:" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Debes instalar Tesseract en Windows:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  1. Descarga:" -ForegroundColor Cyan
        Write-Host "     https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor White
        Write-Host ""
        Write-Host "  2. Instala el .exe (Next, Next, Install)" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  3. Agrega a PATH (opcional pero recomendado):" -ForegroundColor Cyan
        Write-Host "     C:\Program Files\Tesseract-OCR" -ForegroundColor White
        Write-Host ""
        Write-Host "  Alternativa: Si no quieres instalar Tesseract," -ForegroundColor Yellow
        Write-Host "  vuelve a ejecutar este script y elige opción A (EasyOCR)" -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Host "[X] Error en verificación" -ForegroundColor Red
    }
    
} else {
    Write-Host "[X] Opción inválida" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host '  COMO USAR EL DETECTOR CON OCR' -ForegroundColor Cyan
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'COMANDO PRINCIPAL:' -ForegroundColor Yellow
Write-Host '  python mi_detector_ocr.py --modo camara' -ForegroundColor White
Write-Host ''
Write-Host 'FUNCIONALIDADES:' -ForegroundColor Yellow
Write-Host '  Detecta dorsales automaticamente' -ForegroundColor Green
Write-Host '  Lee el NUMERO del dorsal con OCR' -ForegroundColor Green
Write-Host '  Registra en Excel: Posicion | Dorsal | Hora | Observaciones' -ForegroundColor Green
Write-Host '  Evita duplicados' -ForegroundColor Green
Write-Host ''
Write-Host 'COLORES:' -ForegroundColor Yellow
Write-Host '  Verde  - Dorsal nuevo detectado' -ForegroundColor Green
Write-Host '  Naranja - Dorsal ya registrado' -ForegroundColor Yellow
Write-Host '  Rojo   - Dorsal sin numero legible' -ForegroundColor Red
Write-Host ''
Write-Host 'CONTROLES:' -ForegroundColor Yellow
Write-Host "  's' - Ver estadísticas" -ForegroundColor White
Write-Host "  'c' - Capturar imagen" -ForegroundColor White
Write-Host "  'ESPACIO' - Pausar/Reanudar" -ForegroundColor White
Write-Host "  'ESC' o 'q' - Salir" -ForegroundColor White
Write-Host ''
Write-Host '================================================================' -ForegroundColor Cyan
Write-Host ''

Read-Host 'Presiona Enter para finalizar'
