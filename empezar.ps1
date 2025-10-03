# ═══════════════════════════════════════════════════════════════════════════
# SCRIPT DE INICIO RÁPIDO - Todo en Uno
# Ejecuta esto para instalar y/o usar el detector
# ═══════════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "║          🏃 DETECTOR DE NÚMEROS DE DORSAL 🏃                   ║" -ForegroundColor Cyan
Write-Host "║              Script de Inicio Rápido                           ║" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Verificar si el entorno virtual existe
$venvExists = Test-Path ".\venv\Scripts\Activate.ps1"

if (-not $venvExists) {
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host "  PRIMERA VEZ - INSTALACIÓN REQUERIDA" -ForegroundColor Yellow
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "No se detectó instalación previa." -ForegroundColor White
    Write-Host ""
    Write-Host "Necesitas ejecutar primero el instalador." -ForegroundColor White
    Write-Host "Esto tomará 5-10 minutos." -ForegroundColor White
    Write-Host ""
    Write-Host "¿Deseas instalar ahora? (S/N)" -ForegroundColor Cyan
    $respuesta = Read-Host
    
    if ($respuesta -eq "S" -or $respuesta -eq "s" -or $respuesta -eq "Y" -or $respuesta -eq "y" -or $respuesta -eq "") {
        Write-Host ""
        Write-Host "🔄 Iniciando instalación..." -ForegroundColor Green
        Write-Host ""
        
        if (Test-Path ".\instalar_corregido.ps1") {
            & .\instalar_corregido.ps1
        } elseif (Test-Path ".\instalar.ps1") {
            Write-Host "⚠️  Usando instalar.ps1 (no encontré instalar_corregido.ps1)" -ForegroundColor Yellow
            & .\instalar.ps1
        } else {
            Write-Host "❌ ERROR: No se encontró script de instalación" -ForegroundColor Red
            Write-Host ""
            Write-Host "Archivos buscados:" -ForegroundColor Yellow
            Write-Host "  - instalar_corregido.ps1" -ForegroundColor White
            Write-Host "  - instalar.ps1" -ForegroundColor White
            Write-Host ""
            Read-Host "Presiona Enter para salir"
            exit 1
        }
        
        Write-Host ""
        Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
        Write-Host ""
        
        if (Test-Path ".\venv\Scripts\Activate.ps1") {
            Write-Host "✅ Instalación completada!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Presiona Enter para continuar al menú del detector..." -ForegroundColor Cyan
            Read-Host
        } else {
            Write-Host "❌ La instalación falló o fue cancelada" -ForegroundColor Red
            Write-Host ""
            Write-Host "Revisa los mensajes de error arriba" -ForegroundColor Yellow
            Write-Host "Consulta: SOLUCION_GPU_NO_DETECTADA.md o PASOS_COMPLETOS.txt" -ForegroundColor Yellow
            Write-Host ""
            Read-Host "Presiona Enter para salir"
            exit 1
        }
    } else {
        Write-Host ""
        Write-Host "Instalación cancelada." -ForegroundColor Yellow
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

# Si llegamos aquí, el entorno existe
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  ✅ INSTALACIÓN DETECTADA" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""

# Verificar si mi_detector.py existe
if (-not (Test-Path ".\mi_detector.py")) {
    Write-Host "⚠️  ADVERTENCIA: No se encuentra mi_detector.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "El detector propio no está disponible." -ForegroundColor White
    Write-Host "¿Qué deseas hacer?" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1 - Usar Jupyter Notebooks (método original)" -ForegroundColor White
    Write-Host "  2 - Salir" -ForegroundColor White
    Write-Host ""
    $opcion = Read-Host "Selección (1-2)"
    
    if ($opcion -eq "1") {
        Write-Host ""
        Write-Host "🔄 Activando entorno e iniciando Jupyter..." -ForegroundColor Green
        & .\venv\Scripts\Activate.ps1
        Set-Location "notebooks+utils+data"
        jupyter notebook
        exit 0
    } else {
        exit 0
    }
}

# Activar entorno virtual
Write-Host "🔄 Activando entorno virtual..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error al activar entorno virtual" -ForegroundColor Red
    Write-Host ""
    Write-Host "Intenta ejecutar manualmente:" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host "✅ Entorno virtual activado" -ForegroundColor Green
Write-Host ""

# Menú principal
while ($true) {
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  ¿QUÉ DESEAS HACER?" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  INICIO RÁPIDO (Recomendado):" -ForegroundColor Yellow
    Write-Host "  ─────────────────────────────" -ForegroundColor Gray
    Write-Host "  1️⃣  - Test rápido con imagen de ejemplo" -ForegroundColor White
    Write-Host "  2️⃣  - Cámara en tiempo real" -ForegroundColor White
    Write-Host ""
    Write-Host "  OPCIONES COMPLETAS:" -ForegroundColor Yellow
    Write-Host "  ───────────────────" -ForegroundColor Gray
    Write-Host "  3️⃣  - Menú completo del detector (todas las opciones)" -ForegroundColor White
    Write-Host "  4️⃣  - Jupyter Notebooks (método original)" -ForegroundColor White
    Write-Host "  5️⃣  - Verificar instalación" -ForegroundColor White
    Write-Host "  6️⃣  - Ver ayuda y documentación" -ForegroundColor White
    Write-Host ""
    Write-Host "  0️⃣  - Salir" -ForegroundColor White
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    $opcion = Read-Host "Selecciona una opción (0-6)"
    Write-Host ""
    
    switch ($opcion) {
        "1" {
            Write-Host "🖼️  Ejecutando test rápido con imagen de ejemplo..." -ForegroundColor Green
            Write-Host ""
            
            $rutaImagen = "notebooks+utils+data\BibDetectorSample.jpeg"
            
            if (Test-Path $rutaImagen) {
                python mi_detector.py --modo imagen --archivo $rutaImagen
            } else {
                Write-Host "❌ No se encontró la imagen de ejemplo" -ForegroundColor Red
                Write-Host "   Ruta esperada: $rutaImagen" -ForegroundColor Yellow
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "2" {
            Write-Host "🎥 Iniciando detección con cámara..." -ForegroundColor Green
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
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "3" {
            Write-Host "🎯 Abriendo menú completo del detector..." -ForegroundColor Green
            Write-Host ""
            
            if (Test-Path ".\iniciar_detector.ps1") {
                & .\iniciar_detector.ps1
            } else {
                Write-Host "❌ No se encontró iniciar_detector.ps1" -ForegroundColor Red
                Write-Host ""
                Write-Host "Comandos disponibles:" -ForegroundColor Yellow
                Write-Host "  python mi_detector.py --modo camara" -ForegroundColor White
                Write-Host "  python mi_detector.py --modo imagen --archivo ruta" -ForegroundColor White
                Write-Host "  python mi_detector.py --modo video --archivo ruta" -ForegroundColor White
                Write-Host ""
                Write-Host "Ver: python mi_detector.py --help" -ForegroundColor Cyan
                Write-Host ""
                Read-Host "Presiona Enter para volver al menú"
            }
        }
        
        "4" {
            Write-Host "📓 Iniciando Jupyter Notebooks..." -ForegroundColor Green
            Write-Host ""
            Write-Host "Se abrirá tu navegador con Jupyter." -ForegroundColor Cyan
            Write-Host "Recomendado: 05 - Bib Detection Validation & Demo.ipynb" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Presiona Enter para continuar..."
            Read-Host
            Write-Host ""
            
            Set-Location "notebooks+utils+data"
            jupyter notebook
            Set-Location ".."
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "5" {
            Write-Host "🔍 Verificando instalación..." -ForegroundColor Green
            Write-Host ""
            
            if (Test-Path ".\verificar_instalacion.py") {
                python verificar_instalacion.py
            } else {
                Write-Host "⚠️  Script de verificación no encontrado" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "Verificación manual:" -ForegroundColor Cyan
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
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "6" {
            Write-Host "📚 AYUDA Y DOCUMENTACIÓN" -ForegroundColor Green
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "ARCHIVOS DE AYUDA:" -ForegroundColor Yellow
            Write-Host ""
            
            $archivosAyuda = @(
                @{Nombre="PASOS_COMPLETOS.txt"; Desc="Guía paso a paso completa"},
                @{Nombre="USO_MI_DETECTOR.md"; Desc="Manual del detector"},
                @{Nombre="LEEME_MI_DETECTOR.txt"; Desc="Resumen rápido"},
                @{Nombre="COMANDOS_RAPIDOS.ps1"; Desc="Comandos para copiar/pegar"},
                @{Nombre="SOLUCION_GPU_NO_DETECTADA.md"; Desc="Si GPU no funciona"},
                @{Nombre="MANUAL_INSTALACION.md"; Desc="Manual de instalación"}
            )
            
            foreach ($archivo in $archivosAyuda) {
                if (Test-Path $archivo.Nombre) {
                    Write-Host "  ✓ $($archivo.Nombre)" -ForegroundColor Green
                    Write-Host "    $($archivo.Desc)" -ForegroundColor Gray
                } else {
                    Write-Host "  ✗ $($archivo.Nombre)" -ForegroundColor Red
                    Write-Host "    $($archivo.Desc)" -ForegroundColor Gray
                }
                Write-Host ""
            }
            
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "COMANDOS BÁSICOS:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Ver ayuda del detector:" -ForegroundColor White
            Write-Host "  python mi_detector.py --help" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Test con imagen:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo imagen --archivo imagen.jpg" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Cámara en tiempo real:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo camara" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Video:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo video --archivo video.mp4" -ForegroundColor Gray
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            
            Read-Host "Presiona Enter para volver al menú"
        }
        
        "0" {
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "  👋 ¡Hasta pronto!" -ForegroundColor Green
            Write-Host ""
            Write-Host "  Para volver a usar el detector, ejecuta:" -ForegroundColor Cyan
            Write-Host "  .\empezar.ps1" -ForegroundColor White
            Write-Host ""
            Write-Host "  O directamente:" -ForegroundColor Cyan
            Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo camara" -ForegroundColor White
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            exit 0
        }
        
        default {
            Write-Host "❌ Opción inválida. Por favor selecciona 0-6" -ForegroundColor Red
            Write-Host ""
            Start-Sleep -Seconds 2
        }
    }
    
    Write-Host ""
}
