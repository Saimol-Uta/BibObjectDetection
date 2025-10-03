"""
DETECTOR DE DORSALES CON OCR Y REGISTRO EN EXCEL
================================================
Detecta dorsales y lee el número usando OCR
Registra automáticamente en Excel: Posición | Dorsal | HoraLlegada | Observaciones

Autor: Sistema de Detección de Dorsales
Fecha: 2025-10-03
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from registro_llegadas import RegistroLlegadas

# Intentar importar EasyOCR (más preciso pero más pesado)
try:
    import easyocr
    OCR_DISPONIBLE = 'easyocr'
except ImportError:
    OCR_DISPONIBLE = None

# Si no está EasyOCR, intentar pytesseract (más liviano)
if OCR_DISPONIBLE is None:
    try:
        import pytesseract
        OCR_DISPONIBLE = 'tesseract'
    except ImportError:
        OCR_DISPONIBLE = None

print(f"OCR disponible: {OCR_DISPONIBLE}")


def _check_gui_support():
    """Comprueba si la instalación de OpenCV tiene soporte para ventanas (cv2.imshow).

    Devuelve True si cv2.imshow funciona, False en instalaciones headless.
    """
    try:
        # Crear una ventana temporal y mostrar una pequeña imagen
        cv2.namedWindow("__cv_check_gui__", cv2.WINDOW_NORMAL)
        test_img = np.zeros((2, 2, 3), dtype=np.uint8)
        cv2.imshow("__cv_check_gui__", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("__cv_check_gui__")
        return True
    except Exception:
        return False


USE_GUI = _check_gui_support()
print(f"OpenCV GUI disponible: {USE_GUI}")


class Config:
    """Configuración del detector"""
    # Modelo RBNR
    MODELO_RBNR_CFG = "weights-classes/RBNR_custom-yolov4-tiny-detector.cfg"
    MODELO_RBNR_WEIGHTS = "weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights"
    MODELO_RBNR_NAMES = "weights-classes/RBRN_obj.names"
    
    # Parámetros de detección
    INPUT_SIZE = 416
    CONFIANZA_MIN = 0.5
    NMS_THRESHOLD = 0.4
    
    # Parámetros de OCR
    OCR_CONFIANZA_MIN = 0.3  # Confianza mínima para aceptar OCR
    OCR_WHITELIST = '0123456789'  # Solo números
    
    # Registro
    INTERVALO_REGISTRO = 2.0  # Segundos entre registros del mismo dorsal
    
    # Colores (BGR)
    COLOR_BBOX = (0, 255, 0)        # Verde - dorsal detectado
    COLOR_REGISTRADO = (0, 165, 255)  # Naranja - ya registrado
    COLOR_SIN_NUMERO = (0, 0, 255)  # Rojo - no se pudo leer número
    COLOR_BG = (0, 0, 0)            # Negro - fondo texto
    COLOR_TEXT = (255, 255, 255)    # Blanco - texto


class DetectorConOCR:
    """Detector de dorsales con OCR para lectura de números"""
    
    def __init__(self, archivo_excel='registro_llegadas.xlsx', activar_registro=True):
        """
        Inicializa el detector con OCR
        
        Args:
            archivo_excel: Ruta al archivo Excel para registros
            activar_registro: Si False, solo detecta sin registrar
        """
        self.activar_registro = activar_registro
        self.registro = None
        
        if self.activar_registro:
            self.registro = RegistroLlegadas(archivo_excel, permitir_duplicados=False)
            print(f"✓ Sistema de registro activado: {archivo_excel}")
        else:
            print("⚠ Modo sin registro (solo detección)")
        
        # Control de duplicados
        self.ultimas_detecciones = {}
        
        # Cargar modelo
        self._cargar_modelo()
        
        # Inicializar OCR
        self._inicializar_ocr()
    
    def _cargar_modelo(self):
        """Carga el modelo YOLOv4-tiny"""
        print("\n[1/3] Cargando modelo RBNR...")
        
        cfg_path = Path(Config.MODELO_RBNR_CFG)
        weights_path = Path(Config.MODELO_RBNR_WEIGHTS)
        names_path = Path(Config.MODELO_RBNR_NAMES)
        
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config no encontrado: {cfg_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights no encontrado: {weights_path}")
        if not names_path.exists():
            raise FileNotFoundError(f"Names no encontrado: {names_path}")
        
        # Cargar red
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Obtener capas de salida
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        
        if isinstance(unconnected, np.ndarray):
            if len(unconnected.shape) == 1:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
        else:
            self.output_layers = [layer_names[i - 1] for i in unconnected]
        
        # Cargar clases
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print(f"   ✓ Modelo cargado (CPU)")
        print(f"   ✓ Clases: {self.classes}")
    
    def _inicializar_ocr(self):
        """Inicializa el motor OCR"""
        print("\n[2/3] Inicializando OCR...")
        
        if OCR_DISPONIBLE == 'easyocr':
            print("   Usando EasyOCR (alta precisión)...")
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("   ✓ EasyOCR listo")
        elif OCR_DISPONIBLE == 'tesseract':
            print("   Usando Tesseract OCR...")
            # Configurar tesseract para solo números
            self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            print("   ✓ Tesseract listo")
        else:
            print("   ✗ No hay OCR disponible")
            print("\n   INSTALAR OCR:")
            print("     Opción A (recomendado): pip install easyocr")
            print("     Opción B: pip install pytesseract")
            raise ImportError("Necesitas instalar easyocr o pytesseract")
        
        print("[3/3] Sistema listo\n")
    
    def leer_numero_dorsal(self, frame, bbox):
        """
        Lee el número del dorsal usando OCR
        
        Args:
            frame: Imagen completa
            bbox: [x, y, w, h] del dorsal
            
        Returns:
            str: Número detectado o None
        """
        x, y, w, h = bbox
        
        # Extraer región del dorsal
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # Preprocesar para mejorar OCR
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Aumentar contraste
        roi_gray = cv2.equalizeHist(roi_gray)
        
        # Binarización adaptativa
        roi_binary = cv2.adaptiveThreshold(
            roi_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # Intentar OCR
        try:
            if OCR_DISPONIBLE == 'easyocr':
                resultados = self.reader.readtext(roi_binary, detail=1)
                
                # Filtrar solo números
                numeros = []
                for (bbox, texto, conf) in resultados:
                    texto_limpio = ''.join(filter(str.isdigit, texto))
                    if texto_limpio and conf > Config.OCR_CONFIANZA_MIN:
                        numeros.append((texto_limpio, conf))
                
                if numeros:
                    # Retornar el de mayor confianza
                    numeros.sort(key=lambda x: x[1], reverse=True)
                    return numeros[0][0]
            
            elif OCR_DISPONIBLE == 'tesseract':
                texto = pytesseract.image_to_string(
                    roi_binary, 
                    config=self.tesseract_config
                ).strip()
                
                # Limpiar y validar
                texto_limpio = ''.join(filter(str.isdigit, texto))
                if texto_limpio:
                    return texto_limpio
        
        except Exception as e:
            print(f"   Error OCR: {e}")
        
        return None
    
    def detectar(self, frame):
        """
        Detecta dorsales en el frame
        
        Returns:
            list: Lista de detecciones con bbox, confidence, numero
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
                bbox = boxes[i]
                
                # Leer número del dorsal
                numero = self.leer_numero_dorsal(frame, bbox)
                
                detecciones.append({
                    'bbox': bbox,
                    'confidence': confidences[i],
                    'numero': numero
                })
        
        return detecciones
    
    def registrar_deteccion(self, numero):
        """
        Registra un dorsal en Excel
        
        Args:
            numero: Número del dorsal
            
        Returns:
            dict: Info del registro o None
        """
        if not self.activar_registro or not self.registro or not numero:
            return None
        
        tiempo_actual = time.time()
        
        # Verificar cooldown
        if numero in self.ultimas_detecciones:
            tiempo_ultimo = self.ultimas_detecciones[numero]
            if tiempo_actual - tiempo_ultimo < Config.INTERVALO_REGISTRO:
                return None
        
        # Registrar
        resultado = self.registro.registrar_llegada(numero)
        self.ultimas_detecciones[numero] = tiempo_actual
        
        return resultado
    
    def dibujar_detecciones(self, frame, detecciones):
        """Dibuja las detecciones en el frame"""
        for det in detecciones:
            x, y, w, h = det['bbox']
            conf = det['confidence']
            numero = det['numero']
            
            # Determinar color y posición
            posicion = None
            if numero:
                color_bbox = Config.COLOR_BBOX
                if self.activar_registro and self.registro:
                    posicion = self.registro.obtener_posicion(numero)
                    if posicion is not None:
                        color_bbox = Config.COLOR_REGISTRADO
            else:
                color_bbox = Config.COLOR_SIN_NUMERO
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bbox, 2)
            
            # Preparar texto
            if numero:
                if posicion is not None:
                    texto = f"#{numero}: {conf:.2f} [POS: {posicion}]"
                else:
                    texto = f"#{numero}: {conf:.2f}"
            else:
                texto = f"Dorsal: {conf:.2f} [SIN NUMERO]"
            
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
    """Detección en tiempo real con cámara"""
    print("\n" + "="*70)
    print("  DETECTOR CON OCR - MODO CÁMARA")
    print("="*70)
    print("\nBuscando cámara...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[X] Error: No se pudo abrir la cámara")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   ✓ Cámara encontrada")
    print(f"   Resolución: {width}x{height}")
    
    print("\n" + "="*70)
    print("CONTROLES:")
    print("  'q' o 'ESC' - Salir")
    print("  'c' - Capturar frame")
    print("  's' - Ver estadísticas")
    print("  'ESPACIO' - Pausar/Reanudar")
    print("="*70 + "\n")
    
    print("COLORES:")
    print("  🟢 Verde = Dorsal detectado (nuevo)")
    print("  🟠 Naranja = Dorsal ya registrado")
    print("  🔴 Rojo = Dorsal sin número legible")
    print()
    
    # Variables
    pausado = False
    fps_counter = []
    registros_totales = 0
    
    # Crear directorio output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("Iniciando detección con OCR...\n")
    
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
            
            # Registrar dorsales automáticamente
            for det in detecciones:
                if det['numero']:
                    resultado = detector.registrar_deteccion(det['numero'])
                    if resultado and not resultado.get('duplicado', True):
                        registros_totales += 1
                        print(f"✓ Registrado: #{det['numero']} - Posición {resultado['posicion']}")
            
            # Medir FPS
            elapsed = time.time() - start_time
            fps_actual = 1 / elapsed if elapsed > 0 else 0
            fps_counter.append(fps_actual)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            fps_promedio = sum(fps_counter) / len(fps_counter)
            
            # Dibujar detecciones
            frame = detector.dibujar_detecciones(frame, detecciones)
            
            # Mostrar info
            info_text = f"FPS: {fps_promedio:.1f} | Detectados: {len(detecciones)} | Registros: {registros_totales}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if pausado:
                cv2.putText(frame, "PAUSADO", (width//2 - 100, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Mostrar o salvar frame según disponibilidad GUI
        if USE_GUI:
            cv2.imshow('Detector de Dorsales con OCR', frame)
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord(' '):
                pausado = not pausado
                print(f"{'Pausado' if pausado else 'Reanudado'}")
            elif key == ord('c'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"captura_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"✓ Captura guardada: {filename}")
            elif key == ord('s'):
                if detector.activar_registro and detector.registro:
                    stats = detector.registro.obtener_estadisticas()
                    print("\n" + "="*50)
                    print(" ESTADÍSTICAS")
                    print("="*50)
                    print(f"Total llegadas: {stats['total_llegadas']}")
                    if stats['primera_llegada']:
                        print(f"Primera: {stats['primera_llegada']}")
                    if stats['ultima_llegada']:
                        print(f"Última: {stats['ultima_llegada']}")
                    print("="*50 + "\n")
        else:
            # Headless: guardar frames anotados periódicamente y mostrar info en consola
            headless_dir = output_dir / 'headless'
            headless_dir.mkdir(exist_ok=True)

            # Guardar cada N segundos para no llenar disco
            SAVE_INTERVAL = 2.0
            if not hasattr(detectar_camara, '_last_save'):
                detectar_camara._last_save = 0

            now = time.time()
            if now - detectar_camara._last_save >= SAVE_INTERVAL:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = headless_dir / f"frame_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                detectar_camara._last_save = now
                print(f"[HEADLESS] Frame guardado: {filename} | Detectados: {len(detecciones)} | Registros: {registros_totales}")

            # En modo headless no hay manejo de teclas; usar Ctrl+C para salir
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                print('\nDetección interrumpida por usuario (Ctrl+C)')
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Detector finalizado")


def main():
    parser = argparse.ArgumentParser(description='Detector de Dorsales con OCR')
    parser.add_argument('--modo', choices=['camara', 'imagen'], default='camara',
                       help='Modo de detección')
    parser.add_argument('--archivo', type=str, help='Ruta a imagen (modo imagen)')
    parser.add_argument('--excel', type=str, default='registro_llegadas.xlsx',
                       help='Archivo Excel para registros')
    parser.add_argument('--sin-registro', action='store_true',
                       help='Detectar sin registrar')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  DETECTOR DE DORSALES CON OCR")
    print("="*70)
    
    # Crear detector
    detector = DetectorConOCR(
        archivo_excel=args.excel,
        activar_registro=not args.sin_registro
    )
    
    if args.modo == 'camara':
        detectar_camara(detector)
    elif args.modo == 'imagen':
        if not args.archivo:
            print("[X] Error: Debes especificar --archivo para modo imagen")
            return
        
        frame = cv2.imread(args.archivo)
        if frame is None:
            print(f"[X] Error: No se pudo cargar {args.archivo}")
            return
        
        print(f"\nProcesando: {args.archivo}")
        detecciones = detector.detectar(frame)
        
        # Registrar
        for det in detecciones:
            if det['numero']:
                detector.registrar_deteccion(det['numero'])
        
        # Mostrar
        frame = detector.dibujar_detecciones(frame, detecciones)
        cv2.imshow('Detección', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
