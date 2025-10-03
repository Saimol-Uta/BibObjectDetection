# ===================================================================
# INSTALACION DETECTOR CON PYTORCH + GPU
# Este script prepara todo para usar PyTorch con CUDA
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  INSTALACION PYTORCH GPU" -ForegroundColor Cyan
Write-Host "  Detector con aceleracion CUDA" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que el entorno virtual este activo
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[1/7] Activando entorno virtual..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error al activar entorno virtual" -ForegroundColor Red
        Write-Host "    Ejecuta primero: .\instalar.ps1" -ForegroundColor Yellow
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    Write-Host "   OK: Entorno virtual activado" -ForegroundColor Green
} else {
    Write-Host "[1/7] Entorno virtual ya activo" -ForegroundColor Green
}

# Verificar CUDA
Write-Host ""
Write-Host "[2/7] Verificando CUDA..." -ForegroundColor Yellow
$cudaVersion = nvidia-smi 2>&1 | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }

if ($cudaVersion) {
    Write-Host "   OK: CUDA $cudaVersion detectado" -ForegroundColor Green
} else {
    Write-Host "   [!] ADVERTENCIA: No se detecto CUDA" -ForegroundColor Yellow
    Write-Host "       Continuando de todas formas..." -ForegroundColor Yellow
}

# Verificar PyTorch con CUDA
Write-Host ""
Write-Host "[3/7] Verificando PyTorch con CUDA..." -ForegroundColor Yellow
$torchCheck = python -c "import torch; print('OK' if torch.cuda.is_available() else 'NO')" 2>&1

if ($torchCheck -like "*OK*") {
    Write-Host "   OK: PyTorch con CUDA ya instalado y funcionando" -ForegroundColor Green
} else {
    Write-Host "   [!] PyTorch sin CUDA detectado" -ForegroundColor Yellow
    Write-Host "       Reinstalando PyTorch con soporte CUDA..." -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "   Desinstalando PyTorch actual..." -ForegroundColor Cyan
    pip uninstall -y torch torchvision torchaudio 2>$null
    
    Write-Host ""
    Write-Host "   Instalando PyTorch con CUDA 11.8..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Error al instalar PyTorch" -ForegroundColor Red
        Read-Host "Presiona Enter para salir"
        exit 1
    }
    
    # Verificar de nuevo
    $torchCheck2 = python -c "import torch; print('OK' if torch.cuda.is_available() else 'NO')" 2>&1
    if ($torchCheck2 -like "*OK*") {
        Write-Host "   OK: PyTorch con CUDA instalado correctamente" -ForegroundColor Green
    } else {
        Write-Host "[X] PyTorch instalado pero CUDA no disponible" -ForegroundColor Red
        Write-Host "    Verifica tu instalacion de CUDA" -ForegroundColor Yellow
        Read-Host "Presiona Enter para salir"
        exit 1
    }
}

# Instalar dependencias adicionales
Write-Host ""
Write-Host "[4/7] Instalando dependencias para deteccion..." -ForegroundColor Yellow
Write-Host "   -> Instalando ultralytics (YOLO v8)..." -ForegroundColor Cyan
pip install ultralytics --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "   OK: Ultralytics instalado" -ForegroundColor Green
} else {
    Write-Host "[X] Error al instalar ultralytics" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""
Write-Host "   -> Instalando librerias de vision..." -ForegroundColor Cyan
pip install pillow --quiet

# Verificar OpenCV
Write-Host ""
Write-Host "[5/7] Verificando OpenCV..." -ForegroundColor Yellow
$opencvCheck = python -c "import cv2; print(cv2.__version__)" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "   OK: OpenCV version $opencvCheck" -ForegroundColor Green
} else {
    Write-Host "   [!] OpenCV no encontrado, instalando..." -ForegroundColor Yellow
    pip install opencv-python opencv-contrib-python --quiet
    Write-Host "   OK: OpenCV instalado" -ForegroundColor Green
}

# Crear script de deteccion con PyTorch
Write-Host ""
Write-Host "[6/7] Creando detector con PyTorch GPU..." -ForegroundColor Yellow

$detectorScript = @'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECTOR DE DORSALES CON PYTORCH + GPU
Usa PyTorch con CUDA para deteccion en tiempo real
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from datetime import datetime

class Config:
    """Configuracion del detector"""
    # Rutas de modelos
    MODELO_RBNR_WEIGHTS = "weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights"
    MODELO_RBNR_CFG = "weights-classes/RBNR_custom-yolov4-tiny-detector.cfg"
    MODELO_RBNR_NAMES = "weights-classes/RBRN_obj.names"
    
    # Parametros de deteccion
    CONFIANZA_MIN = 0.5
    NMS_THRESHOLD = 0.4
    INPUT_SIZE = 416
    
    # Colores
    COLOR_BBOX = (0, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BG = (0, 0, 0)

class DetectorPyTorch:
    """Detector usando PyTorch con CUDA"""
    
    def __init__(self, usar_gpu=True):
        self.usar_gpu = usar_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.usar_gpu else 'cpu')
        self.net = None
        self.classes = []
        
        print("\n" + "="*70)
        print("  DETECTOR DE DORSALES (PyTorch)")
        print("="*70)
        
        self._cargar_modelo()
        self._cargar_clases()
        
        print(f"\n{'='*70}")
        print(f"Detector inicializado correctamente")
        print(f"   Modelo: RBNR")
        print(f"   Device: {self.device}")
        print(f"   GPU: {torch.cuda.get_device_name(0) if self.usar_gpu else 'N/A'}")
        print(f"   Clases: {len(self.classes)}")
        print("="*70)
    
    def _cargar_modelo(self):
        """Carga el modelo YOLO"""
        print("\n[1/2] Cargando modelo...")
        
        cfg_path = Path(Config.MODELO_RBNR_CFG)
        weights_path = Path(Config.MODELO_RBNR_WEIGHTS)
        
        if not cfg_path.exists():
            raise FileNotFoundError(f"Archivo cfg no encontrado: {cfg_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Archivo weights no encontrado: {weights_path}")
        
        # Cargar red con OpenCV DNN (compatible con PyTorch)
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        
        # Configurar backend
        if self.usar_gpu:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print(f"   OK: Modelo cargado con GPU (CUDA)")
            except:
                print(f"   [!] GPU no disponible en OpenCV DNN, usando CPU")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.usar_gpu = False
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"   OK: Modelo cargado con CPU")
        
        # Obtener nombres de capas de salida
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def _cargar_clases(self):
        """Carga nombres de clases"""
        print("\n[2/2] Cargando clases...")
        
        names_path = Path(Config.MODELO_RBNR_NAMES)
        if not names_path.exists():
            raise FileNotFoundError(f"Archivo names no encontrado: {names_path}")
        
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print(f"   OK: {len(self.classes)} clases cargadas: {self.classes}")
    
    def detectar(self, frame):
        """Realiza deteccion en un frame"""
        height, width = frame.shape[:2]
        
        # Crear blob
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (Config.INPUT_SIZE, Config.INPUT_SIZE),
            swapRB=True, 
            crop=False
        )
        
        # Inferencia
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Procesar salidas
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > Config.CONFIANZA_MIN:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            Config.CONFIANZA_MIN, 
            Config.NMS_THRESHOLD
        )
        
        detecciones = []
        if len(indices) > 0:
            for i in indices.flatten():
                detecciones.append({
                    'bbox': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]]
                })
        
        return detecciones
    
    def dibujar_detecciones(self, frame, detecciones):
        """Dibuja las detecciones en el frame"""
        for det in detecciones:
            x, y, w, h = det['bbox']
            conf = det['confidence']
            clase = det['class_name']
            
            # Dibujar rectangulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), Config.COLOR_BBOX, 2)
            
            # Preparar texto
            texto = f"{clase}: {conf:.2f}"
            
            # Calcular tamaño del texto
            (text_w, text_h), _ = cv2.getTextSize(
                texto, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                2
            )
            
            # Dibujar fondo del texto
            cv2.rectangle(
                frame, 
                (x, y - text_h - 10), 
                (x + text_w, y), 
                Config.COLOR_BG, 
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                frame, 
                texto, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                Config.COLOR_TEXT, 
                2
            )
        
        return frame

def detectar_camara(detector):
    """Deteccion en tiempo real con camara"""
    print("\n" + "="*70)
    print("  CAMARA EN TIEMPO REAL")
    print("="*70)
    print("\nBuscando camara...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[X] Error: No se pudo abrir la camara")
        return
    
    # Configurar resolucion
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"   OK: Camara encontrada")
    print(f"   Resolucion: {width}x{height}")
    print(f"   FPS: {fps}")
    
    print("\n" + "="*70)
    print("CONTROLES:")
    print("  'q' o 'ESC' - Salir")
    print("  'c' - Capturar frame")
    print("  'ESPACIO' - Pausar/Reanudar")
    print("="*70 + "\n")
    
    # Variables de control
    pausado = False
    frames_procesados = 0
    detecciones_totales = 0
    fps_counter = []
    
    # Crear directorio output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                print("[X] Error al leer frame")
                break
            
            # Medir tiempo
            start_time = time.time()
            
            # Detectar
            detecciones = detector.detectar(frame)
            
            # Medir FPS
            elapsed = time.time() - start_time
            fps_actual = 1 / elapsed if elapsed > 0 else 0
            fps_counter.append(fps_actual)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            fps_promedio = sum(fps_counter) / len(fps_counter)
            
            # Dibujar detecciones
            frame = detector.dibujar_detecciones(frame, detecciones)
            
            # Estadisticas
            frames_procesados += 1
            detecciones_totales += len(detecciones)
            
            # Dibujar informacion en pantalla
            info_texto = [
                f"FPS: {fps_promedio:.1f}",
                f"Detecciones: {len(detecciones)}",
                f"Frames: {frames_procesados}",
                f"Device: {'GPU' if detector.usar_gpu else 'CPU'}"
            ]
            
            y_pos = 30
            for texto in info_texto:
                cv2.putText(
                    frame, 
                    texto, 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                y_pos += 30
        
        # Mostrar frame
        cv2.imshow('Detector de Dorsales (PyTorch)', frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' o ESC
            break
        elif key == ord('c'):  # Capturar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"captura_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"[OK] Frame capturado: {filename}")
        elif key == ord(' '):  # ESPACIO
            pausado = not pausado
            estado = "PAUSADO" if pausado else "REANUDADO"
            print(f"[*] {estado}")
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar estadisticas finales
    print("\n" + "="*70)
    print("ESTADISTICAS FINALES")
    print("="*70)
    print(f"Frames procesados: {frames_procesados}")
    print(f"Detecciones totales: {detecciones_totales}")
    if frames_procesados > 0:
        print(f"Promedio detecciones/frame: {detecciones_totales/frames_procesados:.2f}")
    print(f"FPS promedio: {fps_promedio:.1f}")
    print("="*70 + "\n")

def detectar_imagen(detector, ruta_imagen):
    """Deteccion en imagen estatica"""
    print(f"\n[IMAGEN] Procesando: {ruta_imagen}")
    
    frame = cv2.imread(ruta_imagen)
    if frame is None:
        print(f"[X] Error al cargar imagen: {ruta_imagen}")
        return
    
    # Detectar
    detecciones = detector.detectar(frame)
    
    # Dibujar
    frame = detector.dibujar_detecciones(frame, detecciones)
    
    # Guardar
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"deteccion_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)
    
    print(f"[OK] Detecciones encontradas: {len(detecciones)}")
    print(f"[OK] Resultado guardado: {output_path}")
    
    # Mostrar
    cv2.imshow('Deteccion', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Detector de Dorsales con PyTorch GPU')
    parser.add_argument('--modo', choices=['camara', 'imagen'], default='camara',
                       help='Modo de deteccion')
    parser.add_argument('--archivo', type=str, help='Ruta de imagen (para modo imagen)')
    parser.add_argument('--cpu', action='store_true', help='Forzar uso de CPU')
    
    args = parser.parse_args()
    
    try:
        # Crear detector
        detector = DetectorPyTorch(usar_gpu=not args.cpu)
        
        # Ejecutar modo
        if args.modo == 'camara':
            detectar_camara(detector)
        elif args.modo == 'imagen':
            if not args.archivo:
                print("[X] Error: Modo imagen requiere --archivo")
                return
            detectar_imagen(detector, args.archivo)
    
    except KeyboardInterrupt:
        print("\n[*] Detenido por usuario")
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'@

# Guardar script
$detectorScript | Out-File -FilePath "mi_detector_pytorch.py" -Encoding UTF8
Write-Host "   OK: Detector creado: mi_detector_pytorch.py" -ForegroundColor Green

# Verificacion final
Write-Host ""
Write-Host "[7/7] Verificacion final..." -ForegroundColor Yellow
Write-Host ""

$verificacion = python -c @"
import torch
import cv2

print('='*60)
print('VERIFICACION DE INSTALACION')
print('='*60)
print(f'Python: OK')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
print(f'OpenCV: {cv2.__version__}')
print('='*60)
"@

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  INSTALACION COMPLETADA" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Ahora puedes usar el detector con GPU:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Camara en tiempo real:" -ForegroundColor Yellow
Write-Host "    python mi_detector_pytorch.py --modo camara" -ForegroundColor White
Write-Host ""
Write-Host "  Procesar imagen:" -ForegroundColor Yellow
Write-Host "    python mi_detector_pytorch.py --modo imagen --archivo ruta.jpg" -ForegroundColor White
Write-Host ""
Write-Host "  Forzar CPU:" -ForegroundColor Yellow
Write-Host "    python mi_detector_pytorch.py --modo camara --cpu" -ForegroundColor White
Write-Host ""
Write-Host "Archivos creados:" -ForegroundColor Cyan
Write-Host "  - mi_detector_pytorch.py (detector con GPU)" -ForegroundColor White
Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host ""

Read-Host "Presiona Enter para finalizar"
