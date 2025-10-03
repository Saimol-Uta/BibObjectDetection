#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETECTOR DE DORSALES CON REGISTRO DE LLEGADAS
Version optimizada con sistema de registro en Excel
Registra automáticamente cada dorsal detectado
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from datetime import datetime
import re

# Importar sistema de registro
from registro_llegadas import RegistroLlegadas
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    _HAS_TESSERACT = False


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
    
    # Parametros de registro
    INTERVALO_REGISTRO = 2.0  # Segundos entre registros del mismo dorsal
    
    # Colores
    COLOR_BBOX = (0, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BG = (0, 0, 0)
    COLOR_REGISTRADO = (255, 165, 0)  # Naranja para dorsales ya registrados


class DetectorConRegistro:
    """Detector optimizado con sistema de registro de llegadas"""
    
    def __init__(self, activar_registro=True, archivo_excel="registro_llegadas.xlsx"):
        self.net = None
        self.classes = []
        self.output_layers = []
        
        # Sistema de registro
        self.activar_registro = activar_registro
        self.registro = None
        self.ultimas_detecciones = {}  # Evitar registros duplicados inmediatos
        
        print("\n" + "="*70)
        print("  DETECTOR DE DORSALES CON REGISTRO")
        print("="*70)
        
        self._cargar_modelo()
        self._cargar_clases()
        
        if self.activar_registro:
            self.registro = RegistroLlegadas(
                archivo_excel=archivo_excel,
                permitir_duplicados=False
            )
        
        print(f"\n{'='*70}")
        print(f"✓ Detector inicializado correctamente")
        print(f"   Modelo: RBNR (YOLOv4-tiny)")
        print(f"   Backend: OpenCV DNN (CPU optimizado)")
        print(f"   Clases: {len(self.classes)}")
        print(f"   Registro: {'ACTIVADO' if activar_registro else 'DESACTIVADO'}")
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
    
    def registrar_deteccion(self, dorsal):
        """
        Registra una detección en el sistema de registro
        Evita duplicados inmediatos usando cooldown
        
        Args:
            dorsal: Número de dorsal detectado
            
        Returns:
            dict: Info del registro o None si no se registró
        """
        if not self.activar_registro or not self.registro:
            return None
        
        tiempo_actual = time.time()
        
        # Verificar cooldown para evitar registros múltiples inmediatos
        if dorsal in self.ultimas_detecciones:
            tiempo_ultimo = self.ultimas_detecciones[dorsal]
            if tiempo_actual - tiempo_ultimo < Config.INTERVALO_REGISTRO:
                return None  # Muy pronto, ignorar
        
        # Registrar llegada
        resultado = self.registro.registrar_llegada(dorsal)
        
        # Actualizar timestamp
        self.ultimas_detecciones[dorsal] = tiempo_actual
        
        return resultado
    
    def dibujar_detecciones(self, frame, detecciones):
        """Dibuja las detecciones en el frame"""
        for det in detecciones:
            x, y, w, h = det['bbox']
            conf = det['confidence']
            clase = det['class_name']
            
            # Verificar si está registrado
            posicion = None
            color_bbox = Config.COLOR_BBOX
            
            if self.activar_registro and self.registro:
                # Intentar extraer número por OCR y obtener posición real
                numero = self.extraer_numero_por_ocr(frame, (x, y, w, h))
                if numero is not None:
                    posicion = self.registro.obtener_posicion(numero)
                else:
                    posicion = self.registro.obtener_posicion(clase)

                if posicion is not None:
                    color_bbox = Config.COLOR_REGISTRADO
            
            # Dibujar rectangulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bbox, 2)
            
            # Preparar texto
            if posicion is not None:
                texto = f"{clase}: {conf:.2f} [POS: {posicion}]"
            else:
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

    def extraer_numero_por_ocr(self, frame, bbox):
        """
        Extrae texto de la región del bbox y devuelve el primer grupo de dígitos encontrado.
        Usa pytesseract si está disponible, si no, intenta una extracción por umbral y OCR simple
        o devuelve None si no se encuentra.
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # Asegurar coordenadas dentro del frame
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Convertir a gris y preprocesar
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Escalar para mejorar OCR
        scale = 2
        gray = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_LINEAR)
        # Aplicar umbral adaptativo
        gray = cv2.medianBlur(gray, 3)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = ''
        if _HAS_TESSERACT:
            try:
                # Configurar para solo dígitos
                custom_config = r'--psm 6 digits'
                text = pytesseract.image_to_string(th, config=custom_config)
            except Exception:
                text = ''
        else:
            # Fallback: intentar extraer dígitos usando OCR si está instalado en sistema pero no disponible como paquete
            try:
                # intentar llamar a pytesseract si está en PATH
                import subprocess, shlex
                cmd = 'tesseract stdin stdout --psm 6 -c tessedit_char_whitelist=0123456789'
                proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                out, _ = proc.communicate(cv2.imencode('.png', th)[1].tobytes())
                text = out.decode('utf-8') if out else ''
            except Exception:
                text = ''

        # Buscar primer grupo de dígitos
        m = re.search(r"(\d{1,6})", text)
        if m:
            return m.group(1)

        # Si no se encontró texto, intentar con técnicas simples: buscar contornos con forma de dígito
        # (fallback ligero) - aquí devolvemos None
        return None


def detectar_camara(detector):
    """Deteccion en tiempo real con camara"""
    print("\n" + "="*70)
    print("  CAMARA EN TIEMPO REAL - MODO REGISTRO")
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
    print("  'r' - Registrar manualmente dorsal visible")
    print("  's' - Ver estadísticas")
    print("  'ESPACIO' - Pausar/Reanudar")
    print("="*70 + "\n")
    
    # Variables de control
    pausado = False
    frames_procesados = 0
    detecciones_totales = 0
    registros_totales = 0
    fps_counter = []
    
    # Crear directorio output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("Iniciando deteccion con registro automatico...\n")
    print("NOTA: Los dorsales se registran automáticamente al ser detectados")
    print("      Verde = no registrado | Naranja = ya registrado\n")
    
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
            
            # Registrar dorsales detectados automáticamente
            for det in detecciones:
                dorsal = det['class_name']
                # Intentar extraer número del bbox (OCR). Si no hay número, usar la etiqueta de clase
                numero = detector.extraer_numero_por_ocr(frame, det['bbox'])
                dorsal_a_registrar = numero if numero is not None else dorsal
                resultado = detector.registrar_deteccion(dorsal_a_registrar)
                if resultado and not resultado.get('duplicado', True):
                    registros_totales += 1
            
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
                f"Registros: {registros_totales}",
                f"Frames: {frames_procesados}"
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
        cv2.imshow('Detector con Registro de Llegadas', frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' o ESC
            break
        elif key == ord('c'):  # Capturar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"captura_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"[✓] Frame capturado: {filename}")
        elif key == ord('s'):  # Estadísticas
            print("\n" + "="*70)
            if detector.registro:
                stats = detector.registro.obtener_estadisticas()
                if stats:
                    print(f"Total llegadas registradas: {stats['total_llegadas']}")
                    if stats['total_llegadas'] > 0:
                        print(f"Primer lugar: Dorsal {stats['primer_dorsal']} - {stats['primera_hora']}")
                        print(f"Último: Dorsal {stats['ultimo_dorsal']} - {stats['ultima_hora']}")
            print("="*70 + "\n")
        elif key == ord('r'):  # Registro manual
            print("\n[REGISTRO MANUAL]")
            print("Dorsales detectados en pantalla:")
            if detecciones:
                for i, det in enumerate(detecciones, 1):
                    print(f"  {i}. Dorsal: {det['class_name']}")
            else:
                print("  No hay detecciones visibles")
            print()
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
    print(f"Llegadas registradas: {registros_totales}")
    if frames_procesados > 0:
        print(f"Promedio detecciones/frame: {detecciones_totales/frames_procesados:.2f}")
    print(f"FPS promedio: {fps_promedio:.1f}")
    
    if detector.registro:
        print("\n" + "-"*70)
        stats = detector.registro.obtener_estadisticas()
        if stats and stats['total_llegadas'] > 0:
            print(f"Primer lugar: Dorsal {stats['primer_dorsal']}")
            print(f"Última llegada: Dorsal {stats['ultimo_dorsal']}")
            print(f"\nArchivo Excel: registro_llegadas.xlsx")
    
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
    
    # Registrar dorsales detectados
    if detector.activar_registro:
        print("\n[REGISTRO] Dorsales detectados:")
        for det in detecciones:
            dorsal = det['class_name']
            numero = detector.extraer_numero_por_ocr(frame, det['bbox'])
            dorsal_a_registrar = numero if numero is not None else dorsal
            resultado = detector.registrar_deteccion(dorsal_a_registrar)
            if resultado:
                print(f"  ✓ Dorsal {dorsal_a_registrar} - Posición: {resultado['posicion']}")
        print()
    
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
    cv2.imshow('Deteccion con Registro', frame)
    print("\nPresiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Detector de Dorsales con Registro de Llegadas')
    parser.add_argument('--modo', choices=['camara', 'imagen'], default='camara',
                       help='Modo de deteccion')
    parser.add_argument('--archivo', type=str, help='Ruta de imagen (para modo imagen)')
    parser.add_argument('--sin-registro', action='store_true', 
                       help='Desactivar registro automatico')
    parser.add_argument('--excel', type=str, default='registro_llegadas.xlsx',
                       help='Nombre del archivo Excel de registro')
    
    args = parser.parse_args()
    
    try:
        # Crear detector
        detector = DetectorConRegistro(
            activar_registro=not args.sin_registro,
            archivo_excel=args.excel
        )
        
        # Ejecutar modo
        if args.modo == 'camara':
            detectar_camara(detector)
        elif args.modo == 'imagen':
            if not args.archivo:
                print("[X] Error: Modo imagen requiere --archivo")
                print("Ejemplo: python mi_detector_registro.py --modo imagen --archivo imagen.jpg")
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
