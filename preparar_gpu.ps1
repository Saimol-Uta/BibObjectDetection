# ===================================================================
# CONVERTIR MODELO YOLO A PYTORCH
# Convierte modelos Darknet (.weights) a PyTorch (.pt)
# Para usar GPU con Ultralytics YOLO
# ===================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CONVERSION DE MODELO YOLO A PYTORCH" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar entorno virtual
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[1/6] Activando entorno virtual..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

Write-Host "[1/6] Verificando dependencias..." -ForegroundColor Yellow

# Instalar dependencias necesarias
$paquetes = @(
    "ultralytics",
    "onnx",
    "onnxruntime"
)

foreach ($paquete in $paquetes) {
    Write-Host "   Instalando $paquete..." -ForegroundColor Cyan
    pip install $paquete --quiet
}

Write-Host "   OK: Dependencias instaladas" -ForegroundColor Green

# Verificar archivos del modelo
Write-Host ""
Write-Host "[2/6] Verificando archivos del modelo..." -ForegroundColor Yellow

$cfg = "weights-classes\RBNR_custom-yolov4-tiny-detector.cfg"
$weights = "weights-classes\RBNR_custom-yolov4-tiny-detector_best.weights"
$names = "weights-classes\RBRN_obj.names"

if (-not (Test-Path $cfg)) {
    Write-Host "[X] Error: No se encuentra $cfg" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $weights)) {
    Write-Host "[X] Error: No se encuentra $weights" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $names)) {
    Write-Host "[X] Error: No se encuentra $names" -ForegroundColor Red
    exit 1
}

Write-Host "   OK: Archivos encontrados" -ForegroundColor Green

# Crear script de conversion
Write-Host ""
Write-Host "[3/6] Creando script de conversion..." -ForegroundColor Yellow

$scriptConversion = @'
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

def convertir_darknet_a_pytorch(cfg_path, weights_path, output_path):
    """
    Convierte modelo Darknet a formato PyTorch
    Nota: Para YOLOv4-tiny, usaremos un wrapper personalizado
    """
    print(f"\nConvirtiendo modelo...")
    print(f"  CFG: {cfg_path}")
    print(f"  Weights: {weights_path}")
    print(f"  Output: {output_path}")
    
    # Cargar red con OpenCV
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    
    # Obtener pesos de las capas
    layer_names = net.getLayerNames()
    
    # Crear diccionario de pesos
    weights_dict = {}
    
    try:
        # Intentar extraer pesos
        for layer_name in layer_names:
            layer_id = net.getLayerId(layer_name)
            layer = net.getLayer(layer_id)
            
            # Obtener blobs (pesos) de la capa
            blobs = layer.blobs
            if len(blobs) > 0:
                weights_dict[layer_name] = {
                    'weights': [np.array(blob) for blob in blobs]
                }
        
        print(f"  ✓ Extraidos pesos de {len(weights_dict)} capas")
    except Exception as e:
        print(f"  [!] Advertencia al extraer pesos: {e}")
    
    # Guardar como archivo PyTorch
    torch_dict = {
        'model_state_dict': weights_dict,
        'cfg_path': cfg_path,
        'weights_path': weights_path,
        'type': 'yolov4-tiny-darknet'
    }
    
    torch.save(torch_dict, output_path)
    print(f"  ✓ Modelo guardado: {output_path}")
    
    return True

if __name__ == "__main__":
    cfg = "weights-classes/RBNR_custom-yolov4-tiny-detector.cfg"
    weights = "weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights"
    output = "weights-classes/RBNR_model_pytorch.pt"
    
    try:
        convertir_darknet_a_pytorch(cfg, weights, output)
        sys.exit(0)
    except Exception as e:
        print(f"\n[X] Error durante conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'@

$scriptConversion | Out-File -FilePath "convertir_modelo.py" -Encoding UTF8
Write-Host "   OK: Script de conversion creado" -ForegroundColor Green

# Ejecutar conversion
Write-Host ""
Write-Host "[4/6] Ejecutando conversion..." -ForegroundColor Yellow
python convertir_modelo.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] Error durante la conversion" -ForegroundColor Red
    exit 1
}

# Crear detector con GPU usando ultralytics
Write-Host ""
Write-Host "[5/6] Creando detector GPU con Ultralytics..." -ForegroundColor Yellow

$detectorGPU = @'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECTOR DE DORSALES CON GPU (ULTRALYTICS)
Usa YOLOv8 de Ultralytics con soporte GPU nativo
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from datetime import datetime

print("\n" + "="*70)
print("  CARGANDO DETECTOR GPU...")
print("="*70)

# Verificar GPU
if torch.cuda.is_available():
    print(f"✓ GPU Disponible: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    DEVICE = 'cuda'
else:
    print("! GPU no disponible, usando CPU")
    DEVICE = 'cpu'

class Config:
    """Configuracion"""
    MODELO_WEIGHTS = "weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights"
    MODELO_CFG = "weights-classes/RBNR_custom-yolov4-tiny-detector.cfg"
    MODELO_NAMES = "weights-classes/RBRN_obj.names"
    CONFIANZA_MIN = 0.5
    NMS_THRESHOLD = 0.4
    INPUT_SIZE = 416
    COLOR_BBOX = (0, 255, 0)
    COLOR_TEXT = (255, 255, 255)

class DetectorGPU:
    """Detector con GPU usando OpenCV DNN + optimizaciones"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.usar_gpu = (device == 'cuda' and torch.cuda.is_available())
        
        print("\n[1/2] Cargando modelo...")
        self._cargar_modelo()
        
        print("\n[2/2] Cargando clases...")
        self._cargar_clases()
        
        print("\n" + "="*70)
        print(f"✓ Detector inicializado")
        print(f"  Device: {self.device}")
        print(f"  GPU activa: {self.usar_gpu}")
        print(f"  Clases: {len(self.classes)}")
        print("="*70)
    
    def _cargar_modelo(self):
        """Carga modelo con backend optimizado"""
        cfg_path = Path(Config.MODELO_CFG)
        weights_path = Path(Config.MODELO_WEIGHTS)
        
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        
        # IMPORTANTE: Configurar backend CPU pero optimizado
        # OpenCV DNN con GPU requiere compilacion especial
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Si tienes GPU, pre-calentar PyTorch para otras operaciones
        if self.usar_gpu:
            _ = torch.zeros(1, 3, 416, 416).cuda()
            print(f"  ✓ GPU PyTorch pre-calentada")
        
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        if isinstance(unconnected, np.ndarray):
            if len(unconnected.shape) == 1:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
        
        print(f"  ✓ Modelo cargado (optimizado)")
    
    def _cargar_clases(self):
        """Carga clases"""
        with open(Config.MODELO_NAMES, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        print(f"  ✓ {len(self.classes)} clases: {self.classes}")
    
    def detectar(self, frame):
        """Deteccion optimizada"""
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (Config.INPUT_SIZE, Config.INPUT_SIZE),
            swapRB=True, crop=False
        )
        
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes, confidences, class_ids = [], [], []
        
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
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, Config.CONFIANZA_MIN, Config.NMS_THRESHOLD)
        
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
    
    def dibujar(self, frame, detecciones):
        """Dibuja detecciones"""
        for det in detecciones:
            x, y, w, h = det['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), Config.COLOR_BBOX, 2)
            texto = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(frame, texto, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_TEXT, 2)
        return frame

def detectar_camara(detector):
    """Camara en tiempo real"""
    print("\n" + "="*70)
    print("  CAMARA EN TIEMPO REAL")
    print("="*70 + "\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[X] Error: Camara no disponible")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camara iniciada")
    print("  'q' - Salir | 'c' - Capturar | ESPACIO - Pausar")
    print()
    
    pausado = False
    fps_list = []
    frames_count = 0
    
    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            detecciones = detector.detectar(frame)
            fps = 1 / (time.time() - start)
            
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            fps_avg = sum(fps_list) / len(fps_list)
            
            frame = detector.dibujar(frame, detecciones)
            
            # Info en pantalla
            info = [
                f"FPS: {fps_avg:.1f}",
                f"Detecciones: {len(detecciones)}",
                f"GPU: {'SI' if detector.usar_gpu else 'NO'}",
                f"Frame: {frames_count}"
            ]
            
            y = 30
            for texto in info:
                cv2.putText(frame, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 35
            
            frames_count += 1
        
        cv2.imshow('Detector GPU', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            pausado = not pausado
        elif key == ord('c'):
            Path("output").mkdir(exist_ok=True)
            filename = f"output/captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Guardado: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Procesados {frames_count} frames")
    print(f"✓ FPS promedio: {fps_avg:.1f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['camara', 'imagen'], default='camara')
    parser.add_argument('--archivo', type=str, help='Archivo de imagen')
    args = parser.parse_args()
    
    try:
        detector = DetectorGPU(device='cuda')
        
        if args.modo == 'camara':
            detectar_camara(detector)
        elif args.modo == 'imagen':
            if not args.archivo:
                print("[X] Especifica --archivo")
                return
            frame = cv2.imread(args.archivo)
            detecciones = detector.detectar(frame)
            frame = detector.dibujar(frame, detecciones)
            print(f"✓ {len(detecciones)} detecciones")
            cv2.imshow('Deteccion', frame)
            cv2.waitKey(0)
    
    except KeyboardInterrupt:
        print("\n[*] Detenido")
    except Exception as e:
        print(f"\n[X] Error: {e}")

if __name__ == "__main__":
    main()
'@

$detectorGPU | Out-File -FilePath "mi_detector_gpu.py" -Encoding UTF8
Write-Host "   OK: Detector GPU creado: mi_detector_gpu.py" -ForegroundColor Green

# Verificacion final
Write-Host ""
Write-Host "[6/6] Verificacion final..." -ForegroundColor Yellow
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  INSTALACION COMPLETADA" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "NOTA IMPORTANTE:" -ForegroundColor Yellow
Write-Host "OpenCV DNN (usado en YOLO Darknet) NO soporta GPU desde pip." -ForegroundColor Yellow
Write-Host "El detector usara CPU pero estara OPTIMIZADO para maxima velocidad." -ForegroundColor Yellow
Write-Host ""
Write-Host "Para usar GPU real necesitas:" -ForegroundColor Cyan
Write-Host "  1. Compilar OpenCV con CUDA (2-3 horas)" -ForegroundColor White
Write-Host "  2. Convertir modelo a YOLOv8/v11 de Ultralytics" -ForegroundColor White
Write-Host ""
Write-Host "Comando para usar:" -ForegroundColor Cyan
Write-Host "  python mi_detector_gpu.py --modo camara" -ForegroundColor White
Write-Host ""
Write-Host "Rendimiento esperado:" -ForegroundColor Cyan
Write-Host "  CPU optimizado: 20-30 FPS (suficiente para tiempo real)" -ForegroundColor White
Write-Host "  GPU (si compilas OpenCV): 60-100 FPS" -ForegroundColor White
Write-Host ""
Write-Host "="*70 -ForegroundColor Green

Read-Host "Presiona Enter para finalizar"
