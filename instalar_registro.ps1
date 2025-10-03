# ===================================================================
# INSTALAR DEPENDENCIAS PARA REGISTRO DE LLEGADAS
# Instala pandas y openpyxl para manejo de Excel
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  INSTALACION: SISTEMA DE REGISTRO DE LLEGADAS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar entorno virtual
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[1/3] Activando entorno virtual..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error: No se pudo activar entorno virtual" -ForegroundColor Red
        Write-Host "    Ejecuta primero: .\instalar.ps1" -ForegroundColor Yellow
        Read-Host "Presiona Enter para salir"
        exit 1
    }
} else {
    Write-Host "[1/3] Entorno virtual activo" -ForegroundColor Green
}

# Instalar pandas
Write-Host ""
Write-Host "[2/3] Instalando pandas (manejo de datos)..." -ForegroundColor Yellow
pip install pandas --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "   OK: pandas instalado" -ForegroundColor Green
} else {
    Write-Host "[X] Error al instalar pandas" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

# Instalar openpyxl
Write-Host ""
Write-Host "[3/3] Instalando openpyxl (archivos Excel)..." -ForegroundColor Yellow
pip install openpyxl --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "   OK: openpyxl instalado" -ForegroundColor Green
} else {
    Write-Host "[X] Error al instalar openpyxl" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

# Verificar instalación
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  VERIFICANDO INSTALACION" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

python -c "import pandas; import openpyxl; print('✓ pandas:', pandas.__version__); print('✓ openpyxl:', openpyxl.__version__)"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  INSTALACION COMPLETADA" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "ARCHIVOS DISPONIBLES:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1. registro_llegadas.py" -ForegroundColor White
    Write-Host "     Modulo independiente para registro en Excel" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. mi_detector_registro.py" -ForegroundColor White
    Write-Host "     Detector con registro automatico de llegadas" -ForegroundColor Gray
    Write-Host ""
    Write-Host "COMANDOS DISPONIBLES:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Probar sistema de registro:" -ForegroundColor Yellow
    Write-Host "    python registro_llegadas.py" -ForegroundColor White
    Write-Host ""
    Write-Host "  Detector con registro (camara):" -ForegroundColor Yellow
    Write-Host "    python mi_detector_registro.py --modo camara" -ForegroundColor White
    Write-Host ""
    Write-Host "  Detector con registro (imagen):" -ForegroundColor Yellow
    Write-Host "    python mi_detector_registro.py --modo imagen --archivo ruta.jpg" -ForegroundColor White
    Write-Host ""
    Write-Host "  Detector sin registro (solo deteccion):" -ForegroundColor Yellow
    Write-Host "    python mi_detector_registro.py --modo camara --sin-registro" -ForegroundColor White
    Write-Host ""
    Write-Host "  Archivo Excel personalizado:" -ForegroundColor Yellow
    Write-Host "    python mi_detector_registro.py --modo camara --excel mi_carrera.xlsx" -ForegroundColor White
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "FUNCIONALIDADES:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  ✓ Registro automatico al detectar dorsal" -ForegroundColor Green
    Write-Host "  ✓ Evita duplicados" -ForegroundColor Green
    Write-Host "  ✓ Guarda posicion, dorsal y hora de llegada" -ForegroundColor Green
    Write-Host "  ✓ Archivo Excel actualizado en tiempo real" -ForegroundColor Green
    Write-Host "  ✓ Dorsales registrados se muestran en naranja" -ForegroundColor Green
    Write-Host "  ✓ Estadisticas con tecla 's'" -ForegroundColor Green
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[X] Error en verificacion" -ForegroundColor Red
    Write-Host ""
}

Read-Host "Presiona Enter para finalizar"
