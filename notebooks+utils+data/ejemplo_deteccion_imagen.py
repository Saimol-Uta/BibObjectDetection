"""
Script de Ejemplo: Detección de Dorsales en Imagen
Usa este script como base para crear tus propios detectores

Uso:
    python ejemplo_deteccion_imagen.py --imagen ruta/a/tu/imagen.jpg
    python ejemplo_deteccion_imagen.py --imagen VIDEO0433.mp4 --es_video
"""

import cv2
import numpy as np
import argparse
import os
import sys

def cargar_modelo(weights_path, config_path, nombres_path, usar_gpu=True):
    """
    Carga el modelo YOLOv4-tiny
    
    Args:
        weights_path: Ruta al archivo .weights
        config_path: Ruta al archivo .cfg
        nombres_path: Ruta al archivo .names con las clases
        usar_gpu: Si True, intenta usar GPU (CUDA)
    
    Returns:
        net: Modelo cargado
        clases: Lista de nombres de clases
        colores: Colores para visualización
    """
    print("🔄 Cargando modelo...")
    
    # Verificar que los archivos existen
    if not os.path.exists(weights_path):
        print(f"❌ Error: No se encuentra {weights_path}")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"❌ Error: No se encuentra {config_path}")
        sys.exit(1)
    if not os.path.exists(nombres_path):
        print(f"❌ Error: No se encuentra {nombres_path}")
        sys.exit(1)
    
    # Cargar red
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # Configurar backend
    if usar_gpu:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)  # FP16 para RTX 3050
            print("✅ Usando GPU (CUDA) con precisión FP16")
        except Exception as e:
            print(f"⚠️  No se pudo usar GPU, usando CPU: {e}")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✅ Usando CPU")
    
    # Cargar nombres de clases
    with open(nombres_path, 'r') as f:
        clases = [line.strip() for line in f.readlines()]
    
    # Generar colores aleatorios para cada clase
    np.random.seed(42)
    colores = np.random.randint(0, 255, size=(len(clases), 3), dtype=np.uint8)
    
    print(f"✅ Modelo cargado: {len(clases)} clases detectables")
    
    return net, clases, colores


def detectar_imagen(imagen, net, clases, colores, umbral_confianza=0.5, umbral_nms=0.4):
    """
    Realiza detección en una imagen
    
    Args:
        imagen: Imagen OpenCV (BGR)
        net: Modelo cargado
        clases: Lista de clases
        colores: Colores para visualización
        umbral_confianza: Umbral mínimo de confianza (0.0-1.0)
        umbral_nms: Umbral para Non-Maximum Suppression
    
    Returns:
        imagen_resultado: Imagen con detecciones dibujadas
        detecciones: Lista de detecciones (clase, confianza, bbox)
    """
    altura, ancho = imagen.shape[:2]
    
    # Crear blob desde la imagen
    blob = cv2.dnn.blobFromImage(
        imagen, 
        1/255.0,  # Escalar valores de píxeles
        (416, 416),  # Tamaño de entrada
        swapRB=True,  # Convertir BGR a RGB
        crop=False
    )
    
    # Realizar inferencia
    net.setInput(blob)
    nombres_capas = net.getLayerNames()
    capas_salida = [nombres_capas[i - 1] for i in net.getUnconnectedOutLayers()]
    salidas = net.forward(capas_salida)
    
    # Procesar detecciones
    cajas = []
    confianzas = []
    ids_clase = []
    
    for salida in salidas:
        for deteccion in salida:
            puntajes = deteccion[5:]
            id_clase = np.argmax(puntajes)
            confianza = puntajes[id_clase]
            
            if confianza > umbral_confianza:
                # Coordenadas del bounding box
                centro_x = int(deteccion[0] * ancho)
                centro_y = int(deteccion[1] * altura)
                w = int(deteccion[2] * ancho)
                h = int(deteccion[3] * altura)
                
                # Esquina superior izquierda
                x = int(centro_x - w / 2)
                y = int(centro_y - h / 2)
                
                cajas.append([x, y, w, h])
                confianzas.append(float(confianza))
                ids_clase.append(id_clase)
    
    # Non-Maximum Suppression para eliminar detecciones duplicadas
    indices = cv2.dnn.NMSBoxes(cajas, confianzas, umbral_confianza, umbral_nms)
    
    # Dibujar detecciones
    imagen_resultado = imagen.copy()
    detecciones = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = cajas[i]
            clase = clases[ids_clase[i]]
            confianza = confianzas[i]
            color = [int(c) for c in colores[ids_clase[i]]]
            
            # Dibujar rectángulo
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, 2)
            
            # Dibujar etiqueta
            etiqueta = f"{clase}: {confianza:.2%}"
            (ancho_texto, alto_texto), baseline = cv2.getTextSize(
                etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Fondo para el texto
            cv2.rectangle(
                imagen_resultado,
                (x, y - alto_texto - baseline - 5),
                (x + ancho_texto, y),
                color,
                -1
            )
            
            # Texto
            cv2.putText(
                imagen_resultado,
                etiqueta,
                (x, y - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            detecciones.append({
                'clase': clase,
                'confianza': confianza,
                'bbox': (x, y, w, h)
            })
    
    return imagen_resultado, detecciones


def main():
    # Configurar argumentos
    parser = argparse.ArgumentParser(description='Detección de dorsales en imágenes')
    parser.add_argument('--imagen', type=str, default='BibDetectorSample.jpeg',
                       help='Ruta a la imagen de entrada')
    parser.add_argument('--salida', type=str, default='resultado_deteccion.jpg',
                       help='Ruta para guardar la imagen con detecciones')
    parser.add_argument('--modelo', type=str, default='RBNR',
                       choices=['RBNR', 'SVHN'],
                       help='Modelo a usar: RBNR (dorsales) o SVHN (dígitos)')
    parser.add_argument('--umbral', type=float, default=0.5,
                       help='Umbral de confianza (0.0-1.0)')
    parser.add_argument('--cpu', action='store_true',
                       help='Forzar uso de CPU en lugar de GPU')
    
    args = parser.parse_args()
    
    # Rutas de archivos del modelo
    if args.modelo == 'RBNR':
        weights = '../weights-classes/RBNR_custom-yolov4-tiny-detector_best.weights'
        config = '../weights-classes/RBNR_custom-yolov4-tiny-detector.cfg'
        nombres = '../weights-classes/RBRN_obj.names'
        print("📋 Usando modelo RBNR (Detección de Dorsales)")
    else:
        weights = '../weights-classes/SVHN_custom-yolov4-tiny-detector_best.weights'
        config = '../weights-classes/SVHN_custom-yolov4-tiny-detector.cfg'
        nombres = '../weights-classes/SVHN_obj.names'
        print("📋 Usando modelo SVHN (Detección de Dígitos)")
    
    # Cargar modelo
    net, clases, colores = cargar_modelo(weights, config, nombres, usar_gpu=not args.cpu)
    
    # Cargar imagen
    print(f"\n📸 Cargando imagen: {args.imagen}")
    imagen = cv2.imread(args.imagen)
    
    if imagen is None:
        print(f"❌ Error: No se pudo cargar la imagen {args.imagen}")
        sys.exit(1)
    
    print(f"✅ Imagen cargada: {imagen.shape[1]}x{imagen.shape[0]} píxeles")
    
    # Realizar detección
    print(f"\n🔍 Detectando con umbral de confianza: {args.umbral:.2%}")
    imagen_resultado, detecciones = detectar_imagen(
        imagen, net, clases, colores,
        umbral_confianza=args.umbral
    )
    
    # Mostrar resultados
    print(f"\n✨ Detecciones encontradas: {len(detecciones)}")
    for i, det in enumerate(detecciones, 1):
        print(f"  {i}. {det['clase']}: {det['confianza']:.2%} - BBox: {det['bbox']}")
    
    # Guardar imagen resultado
    cv2.imwrite(args.salida, imagen_resultado)
    print(f"\n💾 Resultado guardado en: {args.salida}")
    
    # Mostrar imagen (opcional - comentar si no tienes display)
    try:
        cv2.imshow('Detección de Dorsales', imagen_resultado)
        print("\n👀 Mostrando resultado... (presiona cualquier tecla para cerrar)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\n⚠️  No se pudo mostrar la imagen (modo headless)")
    
    print("\n✅ ¡Detección completada exitosamente!")


if __name__ == '__main__':
    main()
