#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detector de Números de Dorsal - Script Principal
Detección en tiempo real usando cámara o archivos de video/imagen

Uso:
    python mi_detector.py --modo camara
    python mi_detector.py --modo imagen --archivo ruta/imagen.jpg
    python mi_detector.py --modo video --archivo ruta/video.mp4
"""

import cv2
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

class Config:
    """Configuración del detector"""
    
    # Rutas de archivos del modelo RBNR (dorsales)
    WEIGHTS_RBNR = "weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights"
    CONFIG_RBNR = "weights-classes/RBNR_custom-yolov4-tiny-detector.cfg"
    NAMES_RBNR = "weights-classes/RBRN_obj.names"
    
    # Rutas de archivos del modelo SVHN (dígitos)
    WEIGHTS_SVHN = "weights-classes/SVHN_custom-yolov4-tiny-detector_best.weights"
    CONFIG_SVHN = "weights-classes/SVHN_custom-yolov4-tiny-detector.cfg"
    NAMES_SVHN = "weights-classes/SVHN_obj.names"
    
    # Parámetros de detección
    CONFIANZA_MIN = 0.5      # Confianza mínima para considerar una detección
    NMS_THRESHOLD = 0.4      # Umbral para Non-Maximum Suppression
    INPUT_SIZE = 416         # Tamaño de entrada para YOLO (416x416)
    
    # Carpetas de salida
    OUTPUT_DIR = "output"
    OUTPUT_IMAGES = os.path.join(OUTPUT_DIR, "images")
    OUTPUT_VIDEOS = os.path.join(OUTPUT_DIR, "videos")
    
    # Colores para visualización (BGR)
    COLOR_DETECCION = (0, 255, 0)     # Verde
    COLOR_TEXTO = (255, 255, 255)      # Blanco
    COLOR_INFO = (0, 255, 255)         # Amarillo
    
    # Configuración de cámara
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30


# ============================================================================
# CLASE DETECTOR
# ============================================================================

class DetectorDorsales:
    """Detector de números de dorsal usando YOLOv4-tiny"""
    
    def __init__(self, modelo='RBNR', usar_gpu=True):
        """
        Inicializa el detector
        
        Args:
            modelo: 'RBNR' para dorsales o 'SVHN' para dígitos
            usar_gpu: True para usar GPU, False para CPU
        """
        self.modelo = modelo
        self.usar_gpu = usar_gpu
        
        print("="*70)
        print(f"  DETECTOR DE {'DORSALES' if modelo == 'RBNR' else 'DÍGITOS'}")
        print("="*70)
        
        # Cargar configuración
        self._cargar_modelo()
        self._cargar_clases()
        self._configurar_backend()
        
        print(f"\n✅ Detector inicializado correctamente")
        print(f"   Modelo: {self.modelo}")
        print(f"   Backend: {'GPU (CUDA)' if self.usar_gpu else 'CPU'}")
        print(f"   Clases: {len(self.clases)}")
        print("="*70 + "\n")
    
    def _cargar_modelo(self):
        """Carga el modelo YOLOv4-tiny"""
        print("\n[1/3] Cargando modelo...")
        
        # Seleccionar archivos según el modelo
        if self.modelo == 'RBNR':
            weights = Config.WEIGHTS_RBNR
            config = Config.CONFIG_RBNR
        else:
            weights = Config.WEIGHTS_SVHN
            config = Config.CONFIG_SVHN
        
        # Verificar que existen
        if not os.path.exists(weights):
            raise FileNotFoundError(f"❌ No se encuentra: {weights}")
        if not os.path.exists(config):
            raise FileNotFoundError(f"❌ No se encuentra: {config}")
        
        # Cargar red
        try:
            self.net = cv2.dnn.readNetFromDarknet(config, weights)
            size_mb = os.path.getsize(weights) / (1024*1024)
            print(f"   ✓ Modelo cargado ({size_mb:.1f} MB)")
        except Exception as e:
            raise RuntimeError(f"❌ Error al cargar modelo: {e}")
    
    def _cargar_clases(self):
        """Carga los nombres de las clases"""
        print("\n[2/3] Cargando clases...")
        
        names = Config.NAMES_RBNR if self.modelo == 'RBNR' else Config.NAMES_SVHN
        
        if not os.path.exists(names):
            raise FileNotFoundError(f"❌ No se encuentra: {names}")
        
        with open(names, 'r') as f:
            self.clases = [line.strip() for line in f.readlines()]
        
        print(f"   ✓ {len(self.clases)} clases cargadas: {self.clases}")
        
        # Generar colores aleatorios para cada clase
        np.random.seed(42)
        self.colores = np.random.randint(0, 255, size=(len(self.clases), 3), dtype='uint8')
    
    def _configurar_backend(self):
        """Configura el backend (GPU o CPU)"""
        print("\n[3/3] Configurando backend...")
        
        if self.usar_gpu:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                print("   ✓ Backend configurado: GPU (CUDA FP16)")
            except Exception as e:
                print(f"   ⚠ No se pudo usar GPU: {e}")
                print("   ⚠ Usando CPU en su lugar")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.usar_gpu = False
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("   ✓ Backend configurado: CPU")
        
        # Obtener nombres de capas de salida
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detectar(self, frame):
        """
        Realiza detección en un frame
        
        Args:
            frame: Imagen en formato OpenCV (BGR)
            
        Returns:
            detecciones: Lista de diccionarios con información de cada detección
        """
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
                    # Coordenadas del bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, Config.CONFIANZA_MIN, Config.NMS_THRESHOLD)
        
        # Preparar resultado
        detecciones = []
        if len(indices) > 0:
            for i in indices.flatten():
                detecciones.append({
                    'clase': self.clases[class_ids[i]],
                    'confianza': confidences[i],
                    'bbox': boxes[i],
                    'class_id': class_ids[i]
                })
        
        return detecciones
    
    def dibujar_detecciones(self, frame, detecciones):
        """
        Dibuja las detecciones en el frame
        
        Args:
            frame: Imagen en formato OpenCV (BGR)
            detecciones: Lista de detecciones del método detectar()
            
        Returns:
            frame: Imagen con las detecciones dibujadas
        """
        for det in detecciones:
            x, y, w, h = det['bbox']
            clase = det['clase']
            confianza = det['confianza']
            color = [int(c) for c in self.colores[det['class_id']]]
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Preparar etiqueta
            etiqueta = f"{clase}: {confianza:.2%}"
            
            # Calcular tamaño del texto
            (texto_w, texto_h), baseline = cv2.getTextSize(
                etiqueta, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                2
            )
            
            # Dibujar fondo para el texto
            cv2.rectangle(
                frame,
                (x, y - texto_h - baseline - 5),
                (x + texto_w, y),
                color,
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                frame,
                etiqueta,
                (x, y - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                Config.COLOR_TEXTO,
                2
            )
        
        return frame


# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================

def detectar_imagen(detector, ruta_imagen, guardar=True):
    """
    Detecta dorsales en una imagen
    
    Args:
        detector: Instancia de DetectorDorsales
        ruta_imagen: Ruta a la imagen
        guardar: Si True, guarda el resultado
    """
    print(f"\n{'='*70}")
    print(f"  PROCESANDO IMAGEN: {os.path.basename(ruta_imagen)}")
    print(f"{'='*70}\n")
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"❌ Error: No se pudo cargar la imagen {ruta_imagen}")
        return
    
    height, width = imagen.shape[:2]
    print(f"📸 Imagen cargada: {width}x{height} píxeles")
    
    # Detectar
    print("🔍 Detectando...")
    detecciones = detector.detectar(imagen)
    
    print(f"\n✨ Detecciones encontradas: {len(detecciones)}")
    for i, det in enumerate(detecciones, 1):
        print(f"   {i}. {det['clase']}: {det['confianza']:.2%} - BBox: {det['bbox']}")
    
    # Dibujar detecciones
    resultado = detector.dibujar_detecciones(imagen.copy(), detecciones)
    
    # Agregar información
    info_texto = f"Detecciones: {len(detecciones)}"
    cv2.putText(resultado, info_texto, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, Config.COLOR_INFO, 2)
    
    # Guardar
    if guardar:
        os.makedirs(Config.OUTPUT_IMAGES, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"deteccion_{timestamp}.jpg"
        ruta_salida = os.path.join(Config.OUTPUT_IMAGES, nombre_archivo)
        cv2.imwrite(ruta_salida, resultado)
        print(f"\n💾 Resultado guardado: {ruta_salida}")
    
    # Mostrar
    print("\n👀 Mostrando resultado (presiona cualquier tecla para cerrar)...")
    cv2.imshow('Deteccion de Dorsales', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print("✅ Procesamiento completado")
    print(f"{'='*70}\n")


def detectar_video(detector, ruta_video=None, guardar=True):
    """
    Detecta dorsales en video o cámara en tiempo real
    
    Args:
        detector: Instancia de DetectorDorsales
        ruta_video: Ruta al video (None para cámara)
        guardar: Si True, guarda el resultado
    """
    es_camara = ruta_video is None
    
    print(f"\n{'='*70}")
    print(f"  {'CÁMARA EN TIEMPO REAL' if es_camara else f'PROCESANDO VIDEO: {os.path.basename(ruta_video)}'}")
    print(f"{'='*70}\n")
    
    # Abrir video o cámara
    if es_camara:
        print("📹 Buscando cámara...")
        cap = None
        for i in range(5):
            test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    cap = test_cap
                    print(f"   ✓ Cámara encontrada en índice {i}")
                    break
                test_cap.release()
        
        if cap is None:
            print("❌ Error: No se encontró ninguna cámara")
            return
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    else:
        print(f"📹 Abriendo video: {ruta_video}")
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir el video {ruta_video}")
            return
    
    # Obtener propiedades
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not es_camara else Config.CAMERA_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not es_camara else -1
    
    print(f"   ✓ Resolución: {width}x{height}")
    print(f"   ✓ FPS: {fps}")
    if not es_camara:
        print(f"   ✓ Frames totales: {total_frames}")
    
    # Configurar salida
    video_writer = None
    if guardar and not es_camara:
        os.makedirs(Config.OUTPUT_VIDEOS, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"deteccion_{timestamp}.mp4"
        ruta_salida = os.path.join(Config.OUTPUT_VIDEOS, nombre_archivo)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))
        print(f"   ✓ Guardando en: {ruta_salida}")
    
    # Instrucciones
    print(f"\n{'='*70}")
    print("CONTROLES:")
    print("  'q' o 'ESC' - Salir")
    if es_camara:
        print("  'c' - Capturar frame")
        print("  'ESPACIO' - Pausar/Reanudar")
    print(f"{'='*70}\n")
    
    # Variables
    frame_count = 0
    detecciones_totales = 0
    paused = False
    capturas = 0
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    if es_camara:
                        print("❌ Error al capturar frame")
                        break
                    else:
                        print("\n✅ Video procesado completamente")
                        break
                
                frame_count += 1
                
                # Detectar
                detecciones = detector.detectar(frame)
                detecciones_totales += len(detecciones)
                
                # Dibujar detecciones
                frame = detector.dibujar_detecciones(frame, detecciones)
            
            # Información en pantalla
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_count}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_INFO, 2)
            
            info_y += 35
            cv2.putText(frame, f"Detecciones: {len(detecciones) if not paused else 0}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_INFO, 2)
            
            info_y += 35
            if paused:
                cv2.putText(frame, "PAUSADO", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Progreso (solo video)
            if not es_camara and total_frames > 0:
                progreso = (frame_count / total_frames) * 100
                barra_w = width - 20
                barra_h = 20
                barra_x = 10
                barra_y = height - 40
                
                # Fondo de la barra
                cv2.rectangle(frame, (barra_x, barra_y), 
                            (barra_x + barra_w, barra_y + barra_h), (50, 50, 50), -1)
                
                # Progreso
                progreso_w = int((progreso / 100) * barra_w)
                cv2.rectangle(frame, (barra_x, barra_y),
                            (barra_x + progreso_w, barra_y + barra_h), (0, 255, 0), -1)
                
                # Texto
                cv2.putText(frame, f"{progreso:.1f}%", 
                           (barra_x + barra_w + 10, barra_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_INFO, 2)
            
            # Controles
            controles_y = height - 10
            controles_texto = "Q:Salir"
            if es_camara:
                controles_texto += " | C:Capturar | ESPACIO:Pausa"
            cv2.putText(frame, controles_texto, (10, controles_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXTO, 1)
            
            # Guardar frame
            if video_writer is not None and not paused:
                video_writer.write(frame)
            
            # Mostrar
            titulo = 'Deteccion de Dorsales - Camara' if es_camara else 'Deteccion de Dorsales - Video'
            cv2.imshow(titulo, frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' o ESC
                print("\n⏹ Deteniendo...")
                break
            elif key == ord('c') and es_camara:  # Capturar
                os.makedirs(Config.OUTPUT_IMAGES, exist_ok=True)
                capturas += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre = f"captura_{timestamp}.jpg"
                ruta = os.path.join(Config.OUTPUT_IMAGES, nombre)
                cv2.imwrite(ruta, frame)
                print(f"📸 Captura guardada: {nombre}")
            elif key == ord(' ') and es_camara:  # Pausar
                paused = not paused
                print(f"{'⏸ Pausado' if paused else '▶ Reanudado'}")
            
            # Mostrar progreso en consola (cada 30 frames)
            if frame_count % 30 == 0 and not es_camara:
                print(f"⏳ Procesando... {frame_count}/{total_frames} frames ({progreso:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n⚠ Interrumpido por usuario")
    
    finally:
        # Limpiar
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Estadísticas
        print(f"\n{'='*70}")
        print("ESTADÍSTICAS")
        print(f"{'='*70}")
        print(f"Frames procesados: {frame_count}")
        print(f"Detecciones totales: {detecciones_totales}")
        if frame_count > 0:
            print(f"Promedio detecciones/frame: {detecciones_totales/frame_count:.2f}")
        if es_camara and capturas > 0:
            print(f"Capturas guardadas: {capturas}")
        print(f"{'='*70}\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal"""
    
    # Parsear argumentos
    parser = argparse.ArgumentParser(
        description='Detector de Números de Dorsal usando YOLOv4-tiny',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python mi_detector.py --modo camara
  python mi_detector.py --modo imagen --archivo foto.jpg
  python mi_detector.py --modo video --archivo video.mp4
  python mi_detector.py --modo imagen --archivo foto.jpg --modelo SVHN --cpu
        """
    )
    
    parser.add_argument(
        '--modo',
        type=str,
        required=True,
        choices=['camara', 'imagen', 'video'],
        help='Modo de operación: camara, imagen o video'
    )
    
    parser.add_argument(
        '--archivo',
        type=str,
        help='Ruta al archivo de imagen o video (requerido para modos imagen/video)'
    )
    
    parser.add_argument(
        '--modelo',
        type=str,
        default='RBNR',
        choices=['RBNR', 'SVHN'],
        help='Modelo a usar: RBNR (dorsales) o SVHN (dígitos). Default: RBNR'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Forzar uso de CPU en lugar de GPU'
    )
    
    parser.add_argument(
        '--no-guardar',
        action='store_true',
        help='No guardar el resultado'
    )
    
    parser.add_argument(
        '--confianza',
        type=float,
        default=0.5,
        help='Umbral de confianza mínima (0.0-1.0). Default: 0.5'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if args.modo in ['imagen', 'video'] and not args.archivo:
        parser.error(f"El modo '{args.modo}' requiere especificar --archivo")
    
    if args.archivo and not os.path.exists(args.archivo):
        print(f"❌ Error: El archivo '{args.archivo}' no existe")
        return 1
    
    # Ajustar configuración
    Config.CONFIANZA_MIN = args.confianza
    
    # Crear detector
    try:
        detector = DetectorDorsales(
            modelo=args.modelo,
            usar_gpu=not args.cpu
        )
    except Exception as e:
        print(f"\n❌ Error al inicializar detector: {e}")
        return 1
    
    # Ejecutar según modo
    try:
        if args.modo == 'camara':
            detectar_video(detector, None, not args.no_guardar)
        elif args.modo == 'imagen':
            detectar_imagen(detector, args.archivo, not args.no_guardar)
        elif args.modo == 'video':
            detectar_video(detector, args.archivo, not args.no_guardar)
    except Exception as e:
        print(f"\n❌ Error durante la detección: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
