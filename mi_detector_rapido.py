#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECTOR DE DORSALES - VERSION OPTIMIZADA
Usa OpenCV DNN con CPU (optimizado para velocidad)
Compatible con todos los sistemas sin necesidad de compilar OpenCV con CUDA
"""

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
    CONFIANZA_MIN = 0.3
    NMS_THRESHOLD = 0.4
    INPUT_SIZE = 416
    
    # Colores
    COLOR_BBOX = (0, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BG = (0, 0, 0)

class DetectorOptimizado:
    """Detector optimizado con OpenCV DNN (CPU)"""
    
    def __init__(self):
        self.net = None
        self.classes = []
        self.output_layers = []
        
        print("\n" + "="*70)
        print("  DETECTOR DE DORSALES - VERSION OPTIMIZADA")
        print("="*70)
        
        self._cargar_modelo()
        self._cargar_clases()
        
        print(f"\n{'='*70}")
        print(f"✓ Detector inicializado correctamente")
        print(f"   Modelo: RBNR (YOLOv4-tiny)")
        print(f"   Backend: OpenCV DNN (CPU optimizado)")
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
        
        # Cargar red
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        
        # Configurar backend (CPU optimizado)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Obtener nombres de capas de salida
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        
        # Compatibilidad con diferentes versiones de OpenCV
        if isinstance(unconnected, np.ndarray):
            if len(unconnected.shape) == 1:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
        else:
            self.output_layers = [layer_names[i - 1] for i in unconnected]
        
        print(f"   ✓ Modelo cargado (CPU optimizado)")
    
    def _cargar_clases(self):
        """Carga nombres de clases"""
        print("\n[2/2] Cargando clases...")
        
        names_path = Path(Config.MODELO_RBNR_NAMES)
        if not names_path.exists():
            raise FileNotFoundError(f"Archivo names no encontrado: {names_path}")
        
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print(f"   ✓ {len(self.classes)} clases cargadas: {self.classes}")
    
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
                # detection: [center_x, center_y, w, h, objectness, class_scores...]
                objectness = float(detection[4])
                scores = detection[5:]
                if len(scores) == 0:
                    continue
                class_id = int(np.argmax(scores))
                class_score = float(scores[class_id])

                # Usar product of objectness * class_score como confianza final
                confidence = objectness * class_score

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
    
    print(f"   ✓ Camara encontrada")
    print(f"   Resolucion: {width}x{height}")
    print(f"   FPS configurados: {fps}")
    
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
    
    print("Iniciando deteccion... (presiona 'q' para salir)\n")
    
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
                f"Backend: CPU"
            ]
            
            y_pos = 30
            for texto in info_texto:
                # Fondo del texto
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (5, y_pos - 25), (tw + 15, y_pos + 5), (0, 0, 0), -1)
                
                # Texto
                cv2.putText(
                    frame, 
                    texto, 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                y_pos += 35
        
        # Mostrar frame
        cv2.imshow('Detector de Dorsales (CPU Optimizado)', frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' o ESC
            break
        elif key == ord('c'):  # Capturar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"captura_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"[✓] Frame capturado: {filename}")
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
    start_time = time.time()
    detecciones = detector.detectar(frame)
    elapsed = time.time() - start_time
    
    # Dibujar
    frame = detector.dibujar_detecciones(frame, detecciones)
    
    # Guardar
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"deteccion_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)
    
    print(f"[✓] Detecciones encontradas: {len(detecciones)}")
    print(f"[✓] Tiempo de procesamiento: {elapsed:.3f}s")
    print(f"[✓] Resultado guardado: {output_path}")
    
    # Mostrar
    cv2.imshow('Deteccion', frame)
    print("\nPresiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Detector de Dorsales Optimizado')
    parser.add_argument('--modo', choices=['camara', 'imagen'], default='camara',
                       help='Modo de deteccion')
    parser.add_argument('--archivo', type=str, help='Ruta de imagen (para modo imagen)')
    parser.add_argument('--conf', type=float, default=None,
                       help='Umbral de confianza final (objectness * class_score). Por defecto: 0.3')
    
    args = parser.parse_args()
    
    try:
        # Crear detector
        detector = DetectorOptimizado()

        # Si se pasó --conf, actualizar el umbral de confianza
        if args.conf is not None:
            Config.CONFIANZA_MIN = float(args.conf)
        
        # Ejecutar modo
        if args.modo == 'camara':
            detectar_camara(detector)
        elif args.modo == 'imagen':
            if not args.archivo:
                print("[X] Error: Modo imagen requiere --archivo")
                print("Ejemplo: python mi_detector_rapido.py --modo imagen --archivo imagen.jpg")
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
