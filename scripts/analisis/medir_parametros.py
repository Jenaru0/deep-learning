"""
Sistema de Medición de Parámetros Estructurales de Fisuras
===========================================================

Funcionalidades:
1. Inferencia con modelo U-Net entrenado
2. Medición de ancho de fisura (mm)
3. Detección de orientación (Horizontal/Vertical/Diagonal)
4. Estimación de profundidad visual (Superficial/Moderada/Profunda)

Autor: Sistema de Detección de Fisuras
Fecha: Octubre 2024
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
from datetime import datetime

# Configuración de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

# Añadir ruta raíz al path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config import RUTA_MODELO_SEGMENTACION


# ============================================================================
# CONFIGURACIÓN DE GPU Y OPTIMIZACIONES
# ============================================================================

def configurar_gpu():
    """Configura GPU con optimizaciones de memoria."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configurada: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}")
    else:
        print("⚠️ No se detectó GPU, usando CPU")


# ============================================================================
# CARGA DE MODELO U-NET
# ============================================================================

class ModeloSegmentacion:
    """Wrapper para el modelo U-Net de segmentación."""
    
    def __init__(self, ruta_modelo: str = None):
        """
        Inicializa el modelo de segmentación.
        
        Args:
            ruta_modelo: Ruta al archivo .keras del modelo entrenado
        """
        self.ruta_modelo = ruta_modelo or RUTA_MODELO_SEGMENTACION
        self.modelo = None
        self.input_shape = (128, 128, 3)
        
    def cargar(self):
        """Carga el modelo entrenado."""
        if not os.path.exists(self.ruta_modelo):
            raise FileNotFoundError(f"Modelo no encontrado: {self.ruta_modelo}")
        
        print(f"📦 Cargando modelo desde: {self.ruta_modelo}")
        
        # Definir métricas personalizadas necesarias
        def iou_metric(y_true, y_pred):
            y_pred = tf.cast(y_pred > 0.5, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return intersection / (union + 1e-7)
        
        def dice_coefficient(y_true, y_pred):
            y_pred = tf.cast(y_pred > 0.5, tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            return (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
        
        def combined_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            dice_loss = 1 - dice_coefficient(y_true, y_pred)
            return 0.5 * bce + 0.5 * dice_loss
        
        # Cargar modelo con custom objects
        self.modelo = keras.models.load_model(
            self.ruta_modelo,
            custom_objects={
                'iou_metric': iou_metric,
                'dice_coefficient': dice_coefficient,
                'combined_loss': combined_loss
            }
        )
        
        print(f"   ✅ Modelo cargado exitosamente")
        print(f"   📐 Input shape: {self.input_shape}")
        
    def predecir(self, imagen: np.ndarray, umbral: float = 0.5) -> np.ndarray:
        """
        Genera máscara de segmentación para una imagen.
        
        Args:
            imagen: Imagen RGB (altura, ancho, 3)
            umbral: Umbral para binarizar la predicción (0-1)
            
        Returns:
            Máscara binaria (altura, ancho) con valores 0 o 255
        """
        if self.modelo is None:
            self.cargar()
        
        # Guardar tamaño original
        original_shape = imagen.shape[:2]
        
        # Preprocesar imagen
        img_resized = cv2.resize(imagen, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predecir
        pred = self.modelo.predict(img_batch, verbose=0)[0]
        
        # Binarizar y redimensionar a tamaño original
        mask_pred = (pred[:, :, 0] > umbral).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_pred, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        return mask_resized


# ============================================================================
# FUNCIÓN HELPER PARA PREDICCIÓN SIMPLE
# ============================================================================

def predecir_mascara(
    modelo: ModeloSegmentacion,
    imagen,  # Puede ser np.ndarray o str (ruta)
    umbral: float = 0.5
) -> np.ndarray:
    """
    Función helper para predecir máscara de segmentación.
    
    Args:
        modelo: Instancia de ModeloSegmentacion
        imagen: Imagen RGB (H, W, 3) como numpy array, o ruta a archivo de imagen
        umbral: Umbral para binarización (default: 0.5)
        
    Returns:
        Máscara binaria (H, W) con valores 0 o 255
    """
    # Si es una ruta, cargar la imagen
    if isinstance(imagen, (str, Path)):
        from PIL import Image
        img = Image.open(imagen).convert('RGB')
        imagen = np.array(img)
    
    return modelo.predecir(imagen, umbral)


# ============================================================================
# MEDICIÓN DE ANCHO DE FISURA
# ============================================================================

def medir_ancho_fisura(
    mascara: np.ndarray,
    pixeles_por_mm: float = 1.0,
    min_area_pixels: int = 50
) -> Dict:
    """
    Mide el ancho de la fisura usando skeletonización y transformada de distancia.
    
    Algoritmo:
    1. Skeletonizar la máscara para obtener la línea central de la fisura
    2. Aplicar transformada de distancia para medir ancho en cada punto
    3. Convertir píxeles a milímetros
    
    Args:
        mascara: Máscara binaria (0 o 255)
        pixeles_por_mm: Factor de calibración (píxeles/mm). Default: 1.0
        min_area_pixels: Área mínima en píxeles para considerar una fisura
        
    Returns:
        Diccionario con:
            - ancho_promedio_mm: Ancho promedio de la fisura
            - ancho_maximo_mm: Ancho máximo detectado
            - ancho_minimo_mm: Ancho mínimo (excluyendo bordes)
            - area_total_mm2: Área total de la fisura
            - num_regiones: Número de regiones de fisura detectadas
            - mapa_anchos: Matriz con anchos en cada punto del esqueleto
    """
    # Binarizar máscara
    mask_bin = (mascara > 127).astype(np.uint8)
    
    # Filtrar regiones pequeñas (ruido)
    labeled = label(mask_bin)
    regions = regionprops(labeled)
    
    # Filtrar por área mínima
    mask_filtered = np.zeros_like(mask_bin)
    num_regiones = 0
    
    for region in regions:
        if region.area >= min_area_pixels:
            mask_filtered[labeled == region.label] = 1
            num_regiones += 1
    
    if mask_filtered.sum() == 0:
        return {
            'ancho_promedio_mm': 0.0,
            'ancho_maximo_mm': 0.0,
            'ancho_minimo_mm': 0.0,
            'area_total_mm2': 0.0,
            'area_total_px': 0,
            'num_regiones': 0,
            'mapa_anchos': None,
            'skeleton': None
        }
    
    # Skeletonizar la máscara
    skeleton = skeletonize(mask_filtered).astype(np.uint8)
    
    # Transformada de distancia euclidiana
    # Mide la distancia desde cada píxel de fisura hasta el fondo
    dist_transform = distance_transform_edt(mask_filtered)
    
    # Extraer anchos en el esqueleto (multiplicar por 2 porque la distancia es al borde)
    anchos_pixels = dist_transform[skeleton > 0] * 2
    
    if len(anchos_pixels) == 0:
        return {
            'ancho_promedio_mm': 0.0,
            'ancho_maximo_mm': 0.0,
            'ancho_minimo_mm': 0.0,
            'area_total_mm2': 0.0,
            'area_total_px': int(mask_filtered.sum()),
            'num_regiones': num_regiones,
            'mapa_anchos': dist_transform,
            'skeleton': skeleton
        }
    
    # Convertir a milímetros
    ancho_promedio_mm = float(np.mean(anchos_pixels) / pixeles_por_mm)
    ancho_maximo_mm = float(np.max(anchos_pixels) / pixeles_por_mm)
    ancho_minimo_mm = float(np.min(anchos_pixels[anchos_pixels > 1]) / pixeles_por_mm) if len(anchos_pixels[anchos_pixels > 1]) > 0 else 0.0
    
    # Área total
    area_total_px = int(mask_filtered.sum())
    area_total_mm2 = float(area_total_px / (pixeles_por_mm ** 2))
    
    return {
        'ancho_promedio_mm': round(ancho_promedio_mm, 2),
        'ancho_maximo_mm': round(ancho_maximo_mm, 2),
        'ancho_minimo_mm': round(ancho_minimo_mm, 2),
        'area_total_mm2': round(area_total_mm2, 2),
        'area_total_px': area_total_px,
        'num_regiones': num_regiones,
        'mapa_anchos': dist_transform,
        'skeleton': skeleton
    }


# ============================================================================
# DETECCIÓN DE ORIENTACIÓN
# ============================================================================

def detectar_orientacion(mascara: np.ndarray, min_line_length: int = 30) -> Dict:
    """
    Detecta la orientación dominante de la fisura usando transformada de Hough.
    
    Clasificación:
    - Horizontal (H): ángulo entre 0-30° o 150-180°
    - Vertical (V): ángulo entre 60-120°
    - Diagonal (D): ángulo entre 30-60° o 120-150°
    
    Args:
        mascara: Máscara binaria (0 o 255)
        min_line_length: Longitud mínima de línea para considerar (píxeles)
        
    Returns:
        Diccionario con:
            - orientacion: 'H' (Horizontal), 'V' (Vertical), 'D' (Diagonal)
            - angulo_grados: Ángulo promedio en grados (0-180)
            - confianza: Confianza de la detección (0-1)
            - num_lineas: Número de líneas detectadas
            - angulos: Lista de todos los ángulos detectados
    """
    # Binarizar máscara
    mask_bin = (mascara > 127).astype(np.uint8) * 255
    
    # Detectar bordes con Canny
    edges = cv2.Canny(mask_bin, 50, 150, apertureSize=3)
    
    # Transformada de Hough probabilística
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        minLineLength=min_line_length,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return {
            'orientacion': 'Desconocida',
            'angulo_grados': 0.0,
            'confianza': 0.0,
            'num_lineas': 0,
            'angulos': []
        }
    
    # Calcular ángulos de todas las líneas
    angulos = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcular ángulo en grados
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            angulo = 90.0
        else:
            angulo = np.abs(np.arctan(dy / dx) * 180 / np.pi)
        
        angulos.append(angulo)
    
    # Calcular ángulo mediano (más robusto que promedio)
    angulo_mediano = float(np.median(angulos))
    
    # Clasificar orientación
    if (angulo_mediano <= 30) or (angulo_mediano >= 150):
        orientacion = 'Horizontal'
    elif 60 <= angulo_mediano <= 120:
        orientacion = 'Vertical'
    else:
        orientacion = 'Diagonal'
    
    # Calcular confianza basada en consistencia de ángulos
    desviacion_std = float(np.std(angulos))
    confianza = max(0.0, min(1.0, 1.0 - (desviacion_std / 45.0)))
    
    return {
        'orientacion': orientacion,
        'angulo_grados': round(angulo_mediano, 1),
        'confianza': round(confianza, 2),
        'num_lineas': len(lines),
        'angulos': [round(a, 1) for a in angulos]
    }


# ============================================================================
# ESTIMACIÓN DE PROFUNDIDAD VISUAL
# ============================================================================

def estimar_profundidad(
    imagen: np.ndarray,
    mascara: np.ndarray,
    metodo: str = 'intensidad'
) -> Dict:
    """
    Estima la profundidad visual de la fisura basándose en características de intensidad.
    
    NOTA: Esta es una estimación VISUAL basada en contraste, NO es profundidad real.
    Para profundidad real se requiere equipo especializado (láser, ultrasonido, etc.)
    
    Heurística de clasificación por intensidad:
    - Profunda: Intensidad promedio < 80 (muy oscura, sombras profundas)
    - Moderada: Intensidad promedio entre 80-120 (gris medio)
    - Superficial: Intensidad promedio > 120 (clara, poca sombra)
    
    Args:
        imagen: Imagen RGB original
        mascara: Máscara binaria de la fisura
        metodo: Método de estimación ('intensidad' por defecto)
        
    Returns:
        Diccionario con:
            - profundidad_categoria: 'Superficial', 'Moderada', 'Profunda'
            - intensidad_promedio: Intensidad promedio en región de fisura (0-255)
            - contraste: Contraste con respecto al fondo
            - confianza: Confianza de la estimación (0-1)
            - advertencia: Mensaje sobre limitaciones del método
    """
    # Binarizar máscara
    mask_bin = (mascara > 127).astype(np.uint8)
    
    if mask_bin.sum() == 0:
        return {
            'profundidad_categoria': 'Desconocida',
            'intensidad_promedio': 0.0,
            'contraste': 0.0,
            'confianza': 0.0,
            'advertencia': 'No se detectó fisura en la máscara'
        }
    
    # Convertir imagen a escala de grises
    if len(imagen.shape) == 3:
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagen.copy()
    
    # Extraer píxeles de la fisura
    fisura_pixels = gray[mask_bin > 0]
    
    # Extraer píxeles del fondo (región dilatada alrededor de la fisura)
    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask_bin, kernel, iterations=2)
    fondo_mask = (mask_dilated > 0) & (mask_bin == 0)
    fondo_pixels = gray[fondo_mask]
    
    if len(fisura_pixels) == 0:
        return {
            'profundidad_categoria': 'Desconocida',
            'intensidad_promedio': 0.0,
            'contraste': 0.0,
            'confianza': 0.0,
            'advertencia': 'No se encontraron píxeles de fisura'
        }
    
    # Calcular intensidad promedio de la fisura
    intensidad_promedio = float(np.mean(fisura_pixels))
    
    # Calcular contraste con fondo
    if len(fondo_pixels) > 0:
        intensidad_fondo = float(np.mean(fondo_pixels))
        contraste = float(abs(intensidad_fondo - intensidad_promedio))
    else:
        intensidad_fondo = 128.0
        contraste = 0.0
    
    # Clasificar profundidad basándose en intensidad
    if intensidad_promedio < 80:
        profundidad_categoria = 'Profunda'
        confianza = min(1.0, contraste / 100.0)
    elif intensidad_promedio < 120:
        profundidad_categoria = 'Moderada'
        confianza = min(1.0, contraste / 80.0)
    else:
        profundidad_categoria = 'Superficial'
        confianza = min(1.0, contraste / 60.0)
    
    return {
        'profundidad_categoria': profundidad_categoria,
        'intensidad_promedio': round(intensidad_promedio, 1),
        'intensidad_fondo': round(intensidad_fondo, 1) if len(fondo_pixels) > 0 else None,
        'contraste': round(contraste, 1),
        'confianza': round(confianza, 2),
        'advertencia': 'Estimación VISUAL basada en intensidad. NO es profundidad física real.'
    }


# ============================================================================
# ANÁLISIS COMPLETO DE FISURA
# ============================================================================

def analizar_fisura_completo(
    imagen_path: str,
    modelo: ModeloSegmentacion = None,
    pixeles_por_mm: float = 1.0,
    umbral_segmentacion: float = 0.5,
    guardar_visualizacion: bool = True,
    dir_salida: str = None
) -> Dict:
    """
    Realiza análisis completo de una fisura: segmentación + medición de parámetros.
    
    Args:
        imagen_path: Ruta a la imagen a analizar
        modelo: Instancia de ModeloSegmentacion (se crea una nueva si es None)
        pixeles_por_mm: Factor de calibración para conversión de unidades
        umbral_segmentacion: Umbral para binarizar predicción (0-1)
        guardar_visualizacion: Si True, guarda imagen con visualizaciones
        dir_salida: Directorio para guardar resultados
        
    Returns:
        Diccionario con todos los parámetros medidos
    """
    # Cargar imagen
    if not os.path.exists(imagen_path):
        raise FileNotFoundError(f"Imagen no encontrada: {imagen_path}")
    
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    print(f"\n{'='*70}")
    print(f"🔍 ANÁLISIS DE FISURA: {os.path.basename(imagen_path)}")
    print(f"{'='*70}")
    
    # 1. Segmentación
    print("\n📊 Paso 1/4: Segmentación con U-Net...")
    if modelo is None:
        modelo = ModeloSegmentacion()
    
    mascara = modelo.predecir(imagen_rgb, umbral=umbral_segmentacion)
    print(f"   ✅ Máscara generada: {np.sum(mascara > 0)} píxeles de fisura")
    
    # 2. Medición de ancho
    print("\n📏 Paso 2/4: Midiendo ancho de fisura...")
    resultado_ancho = medir_ancho_fisura(mascara, pixeles_por_mm=pixeles_por_mm)
    print(f"   ✅ Ancho promedio: {resultado_ancho['ancho_promedio_mm']} mm")
    print(f"   ✅ Ancho máximo: {resultado_ancho['ancho_maximo_mm']} mm")
    print(f"   ✅ Área total: {resultado_ancho['area_total_mm2']} mm²")
    
    # 3. Detección de orientación
    print("\n🧭 Paso 3/4: Detectando orientación...")
    resultado_orientacion = detectar_orientacion(mascara)
    print(f"   ✅ Orientación: {resultado_orientacion['orientacion']}")
    print(f"   ✅ Ángulo: {resultado_orientacion['angulo_grados']}°")
    print(f"   ✅ Confianza: {resultado_orientacion['confianza']*100:.1f}%")
    
    # 4. Estimación de profundidad
    print("\n🕳️  Paso 4/4: Estimando profundidad visual...")
    resultado_profundidad = estimar_profundidad(imagen_rgb, mascara)
    print(f"   ✅ Profundidad: {resultado_profundidad['profundidad_categoria']}")
    print(f"   ✅ Intensidad: {resultado_profundidad['intensidad_promedio']}/255")
    print(f"   ⚠️  {resultado_profundidad['advertencia']}")
    
    # Compilar resultados
    resultados = {
        'imagen': os.path.basename(imagen_path),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuracion': {
            'pixeles_por_mm': pixeles_por_mm,
            'umbral_segmentacion': umbral_segmentacion
        },
        'ancho': resultado_ancho,
        'orientacion': resultado_orientacion,
        'profundidad': resultado_profundidad
    }
    
    # Guardar visualización si se solicita
    if guardar_visualizacion:
        dir_salida = dir_salida or os.path.join(ROOT_DIR, 'resultados', 'analisis_parametros')
        os.makedirs(dir_salida, exist_ok=True)
        
        vis_path = visualizar_analisis(
            imagen_rgb, mascara, resultados, 
            resultado_ancho.get('skeleton'),
            dir_salida
        )
        resultados['visualizacion_path'] = vis_path
    
    print(f"\n{'='*70}")
    print(f"✅ ANÁLISIS COMPLETADO")
    print(f"{'='*70}\n")
    
    return resultados


# ============================================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================================

def visualizar_analisis(
    imagen: np.ndarray,
    mascara: np.ndarray,
    resultados: Dict,
    skeleton: np.ndarray = None,
    dir_salida: str = None
) -> str:
    """
    Crea visualización completa del análisis de fisura.
    
    Args:
        imagen: Imagen RGB original
        mascara: Máscara de segmentación
        resultados: Diccionario con resultados del análisis
        skeleton: Esqueleto de la fisura (opcional)
        dir_salida: Directorio donde guardar la visualización
        
    Returns:
        Ruta del archivo guardado
    """
    # Crear figura con 4 subplots
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Análisis de Fisura: {resultados['imagen']}", 
                 fontsize=16, fontweight='bold')
    
    # 1. Imagen original
    axes[0, 0].imshow(imagen)
    axes[0, 0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Máscara de segmentación overlay
    overlay = imagen.copy()
    mascara_color = np.zeros_like(imagen)
    mascara_color[:, :, 0] = (mascara > 0) * 255  # Rojo
    overlay = cv2.addWeighted(overlay, 0.7, mascara_color, 0.3, 0)
    
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('Segmentación (Overlay)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Esqueleto + Mapa de anchos
    if skeleton is not None and resultados['ancho'].get('mapa_anchos') is not None:
        mapa_anchos = resultados['ancho']['mapa_anchos']
        
        # Crear visualización de mapa de calor
        plt.subplot(2, 2, 3)
        plt.imshow(imagen, alpha=0.5)
        im = plt.imshow(mapa_anchos, cmap='jet', alpha=0.6, vmin=0, vmax=mapa_anchos.max())
        
        # Superponer esqueleto en blanco
        skeleton_overlay = np.zeros_like(imagen)
        skeleton_overlay[skeleton > 0] = [255, 255, 255]
        plt.imshow(skeleton_overlay, alpha=0.3)
        
        plt.colorbar(im, label='Ancho (píxeles)', fraction=0.046, pad=0.04)
        plt.title('Mapa de Anchos', fontsize=12, fontweight='bold')
        plt.axis('off')
    else:
        axes[1, 0].imshow(mascara, cmap='gray')
        axes[1, 0].set_title('Máscara Binaria', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
    
    # 4. Panel de información
    axes[1, 1].axis('off')
    
    info_text = f"""
PARÁMETROS ESTRUCTURALES

═══════════════════════════════════════

📏 ANCHO DE FISURA:
   • Promedio: {resultados['ancho']['ancho_promedio_mm']} mm
   • Máximo: {resultados['ancho']['ancho_maximo_mm']} mm
   • Mínimo: {resultados['ancho']['ancho_minimo_mm']} mm
   • Área total: {resultados['ancho']['area_total_mm2']} mm²
   • Regiones: {resultados['ancho']['num_regiones']}

🧭 ORIENTACIÓN:
   • Tipo: {resultados['orientacion']['orientacion']}
   • Ángulo: {resultados['orientacion']['angulo_grados']}°
   • Confianza: {resultados['orientacion']['confianza']*100:.1f}%
   • Líneas detectadas: {resultados['orientacion']['num_lineas']}

🕳️  PROFUNDIDAD (VISUAL):
   • Categoría: {resultados['profundidad']['profundidad_categoria']}
   • Intensidad: {resultados['profundidad']['intensidad_promedio']}/255
   • Contraste: {resultados['profundidad']['contraste']}
   • Confianza: {resultados['profundidad']['confianza']*100:.1f}%

═══════════════════════════════════════

⚙️ CONFIGURACIÓN:
   • Calibración: {resultados['configuracion']['pixeles_por_mm']} px/mm
   • Umbral: {resultados['configuracion']['umbral_segmentacion']}
   • Timestamp: {resultados['timestamp']}

⚠️  NOTA: Profundidad es estimación visual,
   NO es medición física real.
"""
    
    axes[1, 1].text(0.05, 0.95, info_text, 
                    transform=axes[1, 1].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Guardar figura
    if dir_salida is None:
        dir_salida = os.path.join(ROOT_DIR, 'resultados', 'analisis_parametros')
    
    os.makedirs(dir_salida, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_base = Path(resultados['imagen']).stem
    output_path = os.path.join(dir_salida, f'analisis_{nombre_base}_{timestamp}.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   💾 Visualización guardada: {output_path}")
    
    return output_path


# ============================================================================
# FUNCIÓN PRINCIPAL PARA PRUEBAS
# ============================================================================

def main():
    """Función principal para pruebas del módulo."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de parámetros estructurales de fisuras')
    parser.add_argument('imagen', type=str, help='Ruta a la imagen a analizar')
    parser.add_argument('--pixeles-por-mm', type=float, default=1.0,
                        help='Factor de calibración (píxeles por milímetro)')
    parser.add_argument('--umbral', type=float, default=0.5,
                        help='Umbral de segmentación (0-1)')
    parser.add_argument('--no-visualizar', action='store_true',
                        help='No guardar visualización')
    parser.add_argument('--dir-salida', type=str, default=None,
                        help='Directorio de salida para resultados')
    
    args = parser.parse_args()
    
    # Configurar GPU
    configurar_gpu()
    
    # Realizar análisis
    resultados = analizar_fisura_completo(
        imagen_path=args.imagen,
        pixeles_por_mm=args.pixeles_por_mm,
        umbral_segmentacion=args.umbral,
        guardar_visualizacion=not args.no_visualizar,
        dir_salida=args.dir_salida
    )
    
    # Guardar resultados en JSON
    if args.dir_salida:
        os.makedirs(args.dir_salida, exist_ok=True)
        json_path = os.path.join(args.dir_salida, f'resultados_{Path(args.imagen).stem}.json')
    else:
        dir_salida = os.path.join(ROOT_DIR, 'resultados', 'analisis_parametros')
        os.makedirs(dir_salida, exist_ok=True)
        json_path = os.path.join(dir_salida, f'resultados_{Path(args.imagen).stem}.json')
    
    # Limpiar objetos numpy del diccionario para JSON
    resultados_json = resultados.copy()
    if 'ancho' in resultados_json:
        resultados_json['ancho'] = {k: v for k, v in resultados_json['ancho'].items() 
                                     if k not in ['mapa_anchos', 'skeleton']}
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resultados_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados JSON guardados: {json_path}")


if __name__ == '__main__':
    main()
