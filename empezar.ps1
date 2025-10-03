# ===================================================================
# SCRIPT DE INICIO RAPIDO - Todo en Uno
# Ejecuta esto para instalar y/o usar el detector
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "                                                                " -ForegroundColor Cyan
Write-Host "          DETECTOR DE NUMEROS DE DORSAL                         " -ForegroundColor Cyan
Write-Host "              Script de Inicio Rapido                           " -ForegroundColor Cyan
Write-Host "                                                                " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si el entorno virtual existe
$venvExists = Test-Path ".\venv\Scripts\Activate.ps1"

if (-not $venvExists) {
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host "  PRIMERA VEZ - INSTALACION REQUERIDA" -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "No se detecto instalacion previa." -ForegroundColor White
    Write-Host ""
    Write-Host "Necesitas ejecutar primero el instalador." -ForegroundColor White
    Write-Host "Esto tomara 5-10 minutos." -ForegroundColor White
    Write-Host ""
    Write-Host "Deseas instalar ahora? (S/N)" -ForegroundColor Cyan
    $respuesta = Read-Host
    
    if ($respuesta -eq "S" -or $respuesta -eq "s" -or $respuesta -eq "Y" -or $respuesta -eq "y" -or $respuesta -eq "") {
        Write-Host ""
        Write-Host "[*] Iniciando instalacion..." -ForegroundColor Green
        Write-Host ""
        
        if (Test-Path ".\instalar_corregido.ps1") {
            & .\instalar_corregido.ps1
        } elseif (Test-Path ".\instalar.ps1") {
            Write-Host "[!] Usando instalar.ps1 (no encontre instalar_corregido.ps1)" -ForegroundColor Yellow
            & .\instalar.ps1
        } else {
            Write-Host "[X] ERROR: No se encontro script de instalacion" -ForegroundColor Red
            Write-Host ""
            Write-Host "Archivos buscados:" -ForegroundColor Yellow
            Write-Host "  - instalar_corregido.ps1" -ForegroundColor White
            Write-Host "  - instalar.ps1" -ForegroundColor White
            Write-Host ""
            Read-Host "Presiona Enter para salir"
            exit 1
        }
        
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor Cyan
        Write-Host ""
        
        if (Test-Path ".\venv\Scripts\Activate.ps1") {
            Write-Host "[OK] Instalacion completada!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Presiona Enter para continuar al menu del detector..." -ForegroundColor Cyan
            Read-Host
        } else {
            Write-Host "[X] La instalacion fallo o fue cancelada" -ForegroundColor Red
            Write-Host ""
            Write-Host "Revisa los mensajes de error arriba" -ForegroundColor Yellow
            Write-Host "Consulta: SOLUCION_GPU_NO_DETECTADA.md o PASOS_COMPLETOS.txt" -ForegroundColor Yellow
            Write-Host ""
            Read-Host "Presiona Enter para salir"
            exit 1
        }
    } else {
        Write-Host ""
        Write-Host "Instalacion cancelada." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Para instalar manualmente, ejecuta:" -ForegroundColor Cyan
        Write-Host "  .\instalar_corregido.ps1" -ForegroundColor White
        Write-Host ""
        Write-Host "O lee: PASOS_COMPLETOS.txt" -ForegroundColor Cyan
        Write-Host ""
        Read-Host "Presiona Enter para salir"
        exit 0
    }
}

# Si llegamos aqui, el entorno existe
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  [OK] INSTALACION DETECTADA" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Verificar si mi_detector.py existe
if (-not (Test-Path ".\mi_detector.py")) {
    Write-Host "[!] ADVERTENCIA: No se encuentra mi_detector.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "El detector propio no esta disponible." -ForegroundColor White
    Write-Host "Que deseas hacer?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1 - Usar Jupyter Notebooks (metodo original)" -ForegroundColor White
    Write-Host "  2 - Salir" -ForegroundColor White
    Write-Host ""
    $opcion = Read-Host "Seleccion (1-2)"
    
    if ($opcion -eq "1") {
        Write-Host ""
        Write-Host "[*] Activando entorno e iniciando Jupyter..." -ForegroundColor Green
        & .\venv\Scripts\Activate.ps1
        Set-Location "notebooks+utils+data"
        jupyter notebook
        exit 0
    } else {
        exit 0
    }
}

# Activar entorno virtual
Write-Host "[*] Activando entorno virtual..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] Error al activar entorno virtual" -ForegroundColor Red
    Write-Host ""
    Write-Host "Intenta ejecutar manualmente:" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host "[OK] Entorno virtual activado" -ForegroundColor Green
Write-Host ""

# Menu principal
while ($true) {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  QUE DESEAS HACER?" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  INICIO RAPIDO (Recomendado):" -ForegroundColor Yellow
    Write-Host "  -----------------------------" -ForegroundColor Gray
    Write-Host "  1 - Test rapido con imagen de ejemplo" -ForegroundColor White
    Write-Host "  2 - Camara en tiempo real" -ForegroundColor White
    Write-Host ""
    Write-Host "  OPCIONES COMPLETAS:" -ForegroundColor Yellow
    Write-Host "  -------------------" -ForegroundColor Gray
    Write-Host "  3 - Menu completo del detector (todas las opciones)" -ForegroundColor White
    Write-Host "  4 - Jupyter Notebooks (metodo original)" -ForegroundColor White
    Write-Host "  5 - Verificar instalacion" -ForegroundColor White
    Write-Host "  6 - Ver ayuda y documentacion" -ForegroundColor White
    Write-Host ""
    Write-Host "  0 - Salir" -ForegroundColor White
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    $opcion = Read-Host "Selecciona una opcion (0-6)"
    Write-Host ""
    
    switch ($opcion) {
        "1" {
            Write-Host "[IMAGEN] Ejecutando test rapido con imagen de ejemplo..." -ForegroundColor Green
            Write-Host ""
            
            $rutaImagen = "notebooks+utils+data\BibDetectorSample.jpeg"
            
            if (Test-Path $rutaImagen) {
                python mi_detector.py --modo imagen --archivo $rutaImagen
            } else {
                Write-Host "[X] No se encontro la imagen de ejemplo" -ForegroundColor Red
                Write-Host "   Ruta esperada: $rutaImagen" -ForegroundColor Yellow
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menu..."
            Read-Host
        }
        
        "2" {
            Write-Host "[CAMARA] Iniciando deteccion con camara..." -ForegroundColor Green
            Write-Host ""
            Write-Host "CONTROLES:" -ForegroundColor Yellow
            Write-Host "  Q o ESC     - Salir" -ForegroundColor White
            Write-Host "  C           - Capturar frame" -ForegroundColor White
            Write-Host "  ESPACIO     - Pausar/Reanudar" -ForegroundColor White
            Write-Host ""
            Write-Host "Presiona Enter para continuar..."
            Read-Host
            Write-Host ""
            
            python mi_detector.py --modo camara
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menu..."
            Read-Host
        }
        
        "3" {
            Write-Host "[MENU] Abriendo menu completo del detector..." -ForegroundColor Green
            Write-Host ""
            
            if (Test-Path ".\iniciar_detector.ps1") {
                & .\iniciar_detector.ps1
            } else {
                Write-Host "[X] No se encontro iniciar_detector.ps1" -ForegroundColor Red
                Write-Host ""
                Write-Host "Comandos disponibles:" -ForegroundColor Yellow
                Write-Host "  python mi_detector.py --modo camara" -ForegroundColor White
                Write-Host "  python mi_detector.py --modo imagen --archivo ruta" -ForegroundColor White
                Write-Host "  python mi_detector.py --modo video --archivo ruta" -ForegroundColor White
                Write-Host ""
                Write-Host "Ver: python mi_detector.py --help" -ForegroundColor Cyan
                Write-Host ""
                Read-Host "Presiona Enter para volver al menu"
            }
        }
        
        "4" {
            Write-Host "[JUPYTER] Iniciando Jupyter Notebooks..." -ForegroundColor Green
            Write-Host ""
            Write-Host "Se abrira tu navegador con Jupyter." -ForegroundColor Cyan
            Write-Host "Recomendado: 05 - Bib Detection Validation and Demo.ipynb" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Presiona Enter para continuar..."
            Read-Host
            Write-Host ""
            
            Set-Location "notebooks+utils+data"
            jupyter notebook
            Set-Location ".."
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menu..."
            Read-Host
        }
        
        "5" {
            Write-Host "[VERIFICAR] Verificando instalacion..." -ForegroundColor Green
            Write-Host ""
            
            if (Test-Path ".\verificar_instalacion.py") {
                python verificar_instalacion.py
            } else {
                Write-Host "[!] Script de verificacion no encontrado" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "Verificacion manual:" -ForegroundColor Cyan
                Write-Host ""
                
                Write-Host "Python:" -ForegroundColor Yellow
                python --version
                
                Write-Host ""
                Write-Host "GPU NVIDIA:" -ForegroundColor Yellow
                nvidia-smi 2>&1 | Select-Object -First 5
                
                Write-Host ""
                Write-Host "PyTorch + CUDA:" -ForegroundColor Yellow
                python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')" 2>&1
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menu..."
            Read-Host
        }
        
        "6" {
            Write-Host "[AYUDA] AYUDA Y DOCUMENTACION" -ForegroundColor Green
            Write-Host ""
            Write-Host "================================================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "ARCHIVOS DE AYUDA:" -ForegroundColor Yellow
            Write-Host ""
            
            $archivosAyuda = @(
                @{Nombre="PASOS_COMPLETOS.txt"; Desc="Guia paso a paso completa"},
                @{Nombre="USO_MI_DETECTOR.md"; Desc="Manual del detector"},
                @{Nombre="LEEME_MI_DETECTOR.txt"; Desc="Resumen rapido"},
                @{Nombre="COMANDOS_RAPIDOS.ps1"; Desc="Comandos para copiar/pegar"},
                @{Nombre="SOLUCION_GPU_NO_DETECTADA.md"; Desc="Si GPU no funciona"},
                @{Nombre="MANUAL_INSTALACION.md"; Desc="Manual de instalacion"}
            )
            
            foreach ($archivo in $archivosAyuda) {
                if (Test-Path $archivo.Nombre) {
                    Write-Host "  [OK] $($archivo.Nombre)" -ForegroundColor Green
                    Write-Host "       $($archivo.Desc)" -ForegroundColor Gray
                } else {
                    Write-Host "  [X] $($archivo.Nombre)" -ForegroundColor Red
                    Write-Host "      $($archivo.Desc)" -ForegroundColor Gray
                }
                Write-Host ""
            }
            
            Write-Host "================================================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "COMANDOS BASICOS:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Ver ayuda del detector:" -ForegroundColor White
            Write-Host "  python mi_detector.py --help" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Test con imagen:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo imagen --archivo imagen.jpg" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Camara en tiempo real:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo camara" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Video:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo video --archivo video.mp4" -ForegroundColor Gray
            Write-Host ""
            Write-Host "================================================================" -ForegroundColor Cyan
            Write-Host ""
            
            Read-Host "Presiona Enter para volver al menu"
        }
        
        "0" {
            Write-Host ""
            Write-Host "================================================================" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  Hasta pronto!" -ForegroundColor Green
            Write-Host ""
            Write-Host "  Para volver a usar el detector, ejecuta:" -ForegroundColor Cyan
            Write-Host "  .\empezar.ps1" -ForegroundColor White
            Write-Host ""
            Write-Host "  O directamente:" -ForegroundColor Cyan
            Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo camara" -ForegroundColor White
            Write-Host ""
            Write-Host "================================================================" -ForegroundColor Cyan
            Write-Host ""
            exit 0
        }
        
        default {
            Write-Host "[X] Opcion invalida. Por favor selecciona 0-6" -ForegroundColor Red
            Write-Host ""
            Start-Sleep -Seconds 2
        }
    }
    
    Write-Host ""
}
