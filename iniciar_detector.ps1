# Script de Inicio Rápido para el Detector Propio
# Menú interactivo para ejecutar el detector fácilmente

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "║        🏃 DETECTOR DE NÚMEROS DE DORSAL 🏃                     ║" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Verificar si el entorno virtual existe
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "❌ ERROR: Entorno virtual no encontrado" -ForegroundColor Red
    Write-Host ""
    Write-Host "Por favor ejecuta primero:" -ForegroundColor Yellow
    Write-Host "  .\instalar_corregido.ps1" -ForegroundColor White
    Write-Host ""
    Read-Host "Presiona Enter para salir"
    exit 1
}

# Activar entorno virtual
Write-Host "🔄 Activando entorno virtual..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error al activar entorno virtual" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host "✅ Entorno virtual activado" -ForegroundColor Green
Write-Host ""

# Verificar que mi_detector.py existe
if (-not (Test-Path ".\mi_detector.py")) {
    Write-Host "❌ ERROR: No se encuentra mi_detector.py" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

# Menú principal
while ($true) {
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  ¿QUÉ DESEAS HACER?" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1️⃣  - Detectar con CÁMARA en tiempo real" -ForegroundColor White
    Write-Host "  2️⃣  - Detectar en IMAGEN (ejemplo incluido)" -ForegroundColor White
    Write-Host "  3️⃣  - Detectar en IMAGEN (tu propia foto)" -ForegroundColor White
    Write-Host "  4️⃣  - Detectar en VIDEO (ejemplo incluido)" -ForegroundColor White
    Write-Host "  5️⃣  - Detectar en VIDEO (tu propio video)" -ForegroundColor White
    Write-Host "  6️⃣  - Ver ayuda y opciones avanzadas" -ForegroundColor White
    Write-Host "  7️⃣  - Verificar instalación" -ForegroundColor White
    Write-Host "  0️⃣  - Salir" -ForegroundColor White
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    $opcion = Read-Host "Selecciona una opción (0-7)"
    Write-Host ""
    
    switch ($opcion) {
        "1" {
            Write-Host "🎥 Iniciando detección con cámara..." -ForegroundColor Green
            Write-Host ""
            Write-Host "CONTROLES:" -ForegroundColor Yellow
            Write-Host "  Q o ESC     - Salir" -ForegroundColor White
            Write-Host "  C           - Capturar frame" -ForegroundColor White
            Write-Host "  ESPACIO     - Pausar/Reanudar" -ForegroundColor White
            Write-Host ""
            Write-Host "Presiona Enter para continuar..."
            Read-Host
            
            python mi_detector.py --modo camara
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "2" {
            Write-Host "🖼️  Procesando imagen de ejemplo..." -ForegroundColor Green
            Write-Host ""
            
            $rutaImagen = "notebooks+utils+data\BibDetectorSample.jpeg"
            
            if (Test-Path $rutaImagen) {
                python mi_detector.py --modo imagen --archivo $rutaImagen
            } else {
                Write-Host "❌ No se encontró la imagen de ejemplo" -ForegroundColor Red
                Write-Host "   Buscada en: $rutaImagen" -ForegroundColor Yellow
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "3" {
            Write-Host "🖼️  Detectar en tu propia imagen" -ForegroundColor Green
            Write-Host ""
            Write-Host "Introduce la ruta completa a tu imagen:" -ForegroundColor Yellow
            Write-Host "(Ejemplo: C:\Users\tuusuario\fotos\imagen.jpg)" -ForegroundColor Gray
            Write-Host ""
            
            $rutaImagen = Read-Host "Ruta"
            
            if (Test-Path $rutaImagen) {
                Write-Host ""
                Write-Host "¿Qué modelo quieres usar?" -ForegroundColor Yellow
                Write-Host "  1 - RBNR (Dorsales completos) - Recomendado" -ForegroundColor White
                Write-Host "  2 - SVHN (Dígitos individuales)" -ForegroundColor White
                Write-Host ""
                $modelo = Read-Host "Selección (1-2)"
                
                $modeloNombre = if ($modelo -eq "2") { "SVHN" } else { "RBNR" }
                
                Write-Host ""
                python mi_detector.py --modo imagen --archivo $rutaImagen --modelo $modeloNombre
            } else {
                Write-Host ""
                Write-Host "❌ No se encontró el archivo: $rutaImagen" -ForegroundColor Red
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "4" {
            Write-Host "🎥 Procesando video de ejemplo..." -ForegroundColor Green
            Write-Host ""
            
            $rutaVideo = "notebooks+utils+data\VIDEO0433.mp4"
            
            if (Test-Path $rutaVideo) {
                Write-Host "Este proceso puede tardar varios minutos..." -ForegroundColor Yellow
                Write-Host "El resultado se guardará en: output\videos\" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "Presiona Enter para continuar..."
                Read-Host
                
                python mi_detector.py --modo video --archivo $rutaVideo
                
                Write-Host ""
                Write-Host "✅ Video procesado!" -ForegroundColor Green
                Write-Host "   Busca el resultado en: output\videos\" -ForegroundColor Cyan
            } else {
                Write-Host "❌ No se encontró el video de ejemplo" -ForegroundColor Red
                Write-Host "   Buscado en: $rutaVideo" -ForegroundColor Yellow
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "5" {
            Write-Host "🎥 Detectar en tu propio video" -ForegroundColor Green
            Write-Host ""
            Write-Host "Introduce la ruta completa a tu video:" -ForegroundColor Yellow
            Write-Host "(Ejemplo: C:\Users\tuusuario\videos\carrera.mp4)" -ForegroundColor Gray
            Write-Host ""
            
            $rutaVideo = Read-Host "Ruta"
            
            if (Test-Path $rutaVideo) {
                Write-Host ""
                Write-Host "Este proceso puede tardar varios minutos..." -ForegroundColor Yellow
                Write-Host "El resultado se guardará en: output\videos\" -ForegroundColor Cyan
                Write-Host ""
                Write-Host "Presiona Enter para continuar..."
                Read-Host
                
                python mi_detector.py --modo video --archivo $rutaVideo
                
                Write-Host ""
                Write-Host "✅ Video procesado!" -ForegroundColor Green
                Write-Host "   Busca el resultado en: output\videos\" -ForegroundColor Cyan
            } else {
                Write-Host ""
                Write-Host "❌ No se encontró el archivo: $rutaVideo" -ForegroundColor Red
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "6" {
            Write-Host "📚 AYUDA Y OPCIONES AVANZADAS" -ForegroundColor Green
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "COMANDOS MANUALES:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Cámara:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo camara" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Imagen:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo imagen --archivo ruta\imagen.jpg" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Video:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo video --archivo ruta\video.mp4" -ForegroundColor Gray
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "OPCIONES ADICIONALES:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  --modelo SVHN        Usar modelo de dígitos" -ForegroundColor White
            Write-Host "  --cpu                Forzar uso de CPU" -ForegroundColor White
            Write-Host "  --confianza 0.7      Cambiar umbral (0.0-1.0)" -ForegroundColor White
            Write-Host "  --no-guardar         No guardar resultado" -ForegroundColor White
            Write-Host "  --help               Ver ayuda completa" -ForegroundColor White
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "EJEMPLOS:" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Cámara con modelo de dígitos:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo camara --modelo SVHN" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Imagen con umbral alto (más estricto):" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo imagen --archivo foto.jpg --confianza 0.7" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Video sin usar GPU:" -ForegroundColor White
            Write-Host "  python mi_detector.py --modo video --archivo video.mp4 --cpu" -ForegroundColor Gray
            Write-Host ""
            Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Para más información, lee: USO_MI_DETECTOR.md" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "7" {
            Write-Host "🔍 Verificando instalación..." -ForegroundColor Green
            Write-Host ""
            
            if (Test-Path ".\verificar_instalacion.py") {
                python verificar_instalacion.py
            } else {
                Write-Host "⚠️  Script de verificación no encontrado" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "Verificación manual:" -ForegroundColor Cyan
                Write-Host ""
                
                # Verificar Python
                Write-Host "Python:" -ForegroundColor Yellow
                python --version
                
                # Verificar GPU
                Write-Host ""
                Write-Host "GPU NVIDIA:" -ForegroundColor Yellow
                nvidia-smi | Select-Object -First 5
                
                # Verificar PyTorch
                Write-Host ""
                Write-Host "PyTorch + CUDA:" -ForegroundColor Yellow
                python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
                
                # Verificar archivos del modelo
                Write-Host ""
                Write-Host "Archivos del modelo:" -ForegroundColor Yellow
                if (Test-Path "weights-classes\RBNR_custom-yolov4-tiny-detector_best.weights") {
                    Write-Host "  ✓ RBNR weights" -ForegroundColor Green
                } else {
                    Write-Host "  ✗ RBNR weights" -ForegroundColor Red
                }
                
                if (Test-Path "weights-classes\RBNR_custom-yolov4-tiny-detector.cfg") {
                    Write-Host "  ✓ RBNR config" -ForegroundColor Green
                } else {
                    Write-Host "  ✗ RBNR config" -ForegroundColor Red
                }
            }
            
            Write-Host ""
            Write-Host "Presiona Enter para volver al menú..."
            Read-Host
        }
        
        "0" {
            Write-Host ""
            Write-Host "👋 ¡Hasta pronto!" -ForegroundColor Cyan
            Write-Host ""
            exit 0
        }
        
        default {
            Write-Host "❌ Opción inválida. Por favor selecciona 0-7" -ForegroundColor Red
            Write-Host ""
            Start-Sleep -Seconds 2
        }
    }
    
    Write-Host ""
}
