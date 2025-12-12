"""
Aplicación Web para Detección y Análisis de Fisuras en Estructuras
===================================================================

Interfaz gráfica desarrollada con Streamlit que ofrece dos modos:
1. Detección: Clasificación binaria (fisura/no fisura) con MobileNetV2
2. Segmentación: Análisis detallado con U-Net + medición de parámetros estructurales

Autor: Sistema de Detección de Fisuras
Fecha: Octubre 2025
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io
import sys
import os
from pathlib import Path

# ==========================================
# CONFIGURACIÓN PARA CLOUD (sin GPU)
# ==========================================
if 'STREAMLIT_SHARING' in os.environ or 'DYNO' in os.environ:
    # Forzar CPU en entornos cloud
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configurar el path para importar módulos del proyecto
PROYECTO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROYECTO_ROOT))

# ==========================================
# DESCARGAR MODELOS DESDE GOOGLE DRIVE (OPCIONAL)
# ==========================================
# Solo se ejecuta si los modelos no existen localmente
try:
    from app_web.download_models import download_models
    
    # Usar rutas absolutas relativas al proyecto
    modelo_det_path = PROYECTO_ROOT / 'modelos' / 'deteccion' / 'modelo_deteccion_final.keras'
    modelo_seg_path = PROYECTO_ROOT / 'modelos' / 'segmentacion' / 'unet_segmentacion_final.keras'
    
    modelo_det_existe = modelo_det_path.exists()
    modelo_seg_existe = modelo_seg_path.exists()
    
    if not modelo_det_existe or not modelo_seg_existe:
        with st.spinner("📥 Descargando modelos desde Google Drive..."):
            download_models()
            # Re-verificar después de la descarga
            modelo_det_existe = modelo_det_path.exists()
            modelo_seg_existe = modelo_seg_path.exists()
    
    if modelo_det_existe and modelo_seg_existe:
        st.success("✅ Modelos encontrados localmente")
    elif modelo_det_existe:
        st.success("✅ Modelo de detección encontrado")
        st.warning("⚠️ Modelo de segmentación no disponible")
    else:
        st.error("❌ Error: modelos no descargados correctamente")
except Exception as e:
    # Si falla la descarga, continuar (los modelos pueden estar localmente)
    st.warning(f"⚠️ Error en descarga automática: {e}")
    st.info("Intentando usar modelos locales...")
    pass

# Usar rutas relativas al proyecto en lugar de config.py
# Esto funciona tanto en local como en Streamlit Cloud
MODELOS_DIR = PROYECTO_ROOT / "modelos" / "deteccion"
RUTA_MODELO_SEGMENTACION = PROYECTO_ROOT / "modelos" / "segmentacion" / "unet_segmentacion_final.keras"
IMG_SIZE = 224

# Importar módulos de análisis de parámetros
try:
    from scripts.analisis.medir_parametros import (
        ModeloSegmentacion,
        medir_ancho_fisura,
        detectar_orientacion,
        estimar_profundidad
    )
    SEGMENTACION_DISPONIBLE = True
except ImportError as e:
    SEGMENTACION_DISPONIBLE = False
    print(f"⚠️ Módulo de segmentación no disponible: {e}")


# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Detector de Fisuras",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def calcular_hash_imagen(imagen_array: np.ndarray) -> str:
    """Calcula hash MD5 de imagen para usar en caché."""
    import hashlib
    return hashlib.md5(imagen_array.tobytes()).hexdigest()


@st.cache_resource(show_spinner="⏳ Paso 1/2: Cargando MobileNetV2 (14MB, ~20-30s)...")
def cargar_modelo():
    """
    Carga el modelo entrenado.
    Usa cache de Streamlit para cargar solo una vez.
    
    Returns:
        tensorflow.keras.Model: Modelo cargado
    """
    # Silenciar logs de TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # MODELOS_DIR ya es un Path object
    modelos_dir = MODELOS_DIR if isinstance(MODELOS_DIR, Path) else Path(MODELOS_DIR)
    
    # Buscar modelos en orden de prioridad
    patrones = [
        "modelo_deteccion_final.keras",
        "modelo_deteccion_final.h5",
        "best_model_stage2_*.keras",
        "best_model_stage2_*.h5",
        "best_model_*.keras",
        "best_model_*.h5"
    ]
    
    modelo_path = None
    for patron in patrones:
        archivos = list(modelos_dir.glob(patron))
        if archivos:
            # Si hay múltiples, tomar el más reciente
            modelo_path = max(archivos, key=lambda p: p.stat().st_mtime)
            break
    
    if modelo_path is None:
        st.error("❌ No se encontró ningún modelo entrenado.")
        st.info(f"📁 Buscando en: {modelos_dir}")
        st.info(f"📂 Archivos disponibles: {list(modelos_dir.glob('*')) if modelos_dir.exists() else 'Directorio no existe'}")
        st.info("💡 El modelo debería descargarse automáticamente. Recarga la página si acabas de iniciar la app.")
        st.stop()
    
    try:
        # Cargar sin compilar (más rápido y evita warnings)
        with tf.keras.utils.custom_object_scope({}):
            modelo = tf.keras.models.load_model(modelo_path, compile=False)
        
        # Compilar manualmente para evitar warnings del optimizer
        modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False
        )
        return modelo, modelo_path.name
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.info(f"📁 Modelo buscado: {modelo_path}")
        st.stop()


@st.cache_resource(show_spinner="⏳ Paso 2/2: Cargando U-Net (~10-15s)...")
def cargar_modelo_segmentacion():
    """
    Carga el modelo U-Net para segmentación de fisuras.
    Usa cache de Streamlit para cargar solo una vez.
    
    Returns:
        ModeloSegmentacion: Instancia del modelo de segmentación
    """
    if not SEGMENTACION_DISPONIBLE:
        st.error("❌ Módulo de segmentación no disponible")
        return None
    
    try:
        modelo_seg = ModeloSegmentacion(RUTA_MODELO_SEGMENTACION)
        modelo_seg.cargar()
        return modelo_seg
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo de segmentación: {e}")
        return None


def preprocesar_imagen(imagen_pil, img_size=IMG_SIZE):
    """
    Preprocesa una imagen PIL para el modelo.
    
    Args:
        imagen_pil (PIL.Image): Imagen a preprocesar
        img_size (int or tuple): Tamaño objetivo (si es int, se usa para height y width)
    
    Returns:
        np.ndarray: Imagen preprocesada lista para predicción
    """
    # Convertir a RGB si es necesario
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    
    # Manejar img_size como int o tuple
    if isinstance(img_size, int):
        target_size = (img_size, img_size)
    else:
        target_size = (img_size[1], img_size[0])  # PIL usa (width, height)
    
    # Redimensionar
    imagen_resized = imagen_pil.resize(target_size)
    
    # Convertir a array numpy y normalizar
    img_array = np.array(imagen_resized, dtype=np.float32)
    img_array = img_array / 255.0  # Normalizar a [0, 1]
    
    # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predecir(modelo, imagen_preprocesada, umbral=0.5):
    """
    Realiza la predicción sobre una imagen.
    
    IMPORTANTE: El modelo fue entrenado con ImageDataGenerator donde:
    - class_indices = {'cracked': 0, 'uncracked': 1}
    - En modo binary, la salida es la probabilidad de la clase con índice 1 (uncracked)
    - Por lo tanto, necesitamos INVERTIR: prob_uncracked → prob_cracked
    
    Args:
        modelo: Modelo de Keras
        imagen_preprocesada: Imagen preprocesada (batch, height, width, channels)
        umbral: Umbral de decisión (default: 0.5)
    
    Returns:
        tuple: (clase_predicha, confianza, probabilidad_cracked)
    """
    # Realizar predicción
    prediccion = modelo.predict(imagen_preprocesada, verbose=0)
    prob_uncracked = float(prediccion[0][0])  # El modelo predice probabilidad de UNCRACKED
    
    # INVERTIR: convertir probabilidad de uncracked a probabilidad de cracked
    prob_cracked = 1.0 - prob_uncracked
    
    # Clasificar según el umbral
    if prob_cracked >= umbral:
        clase = "FISURA DETECTADA"
        confianza = prob_cracked
    else:
        clase = "SIN FISURA"
        confianza = 1.0 - prob_cracked
    
    return clase, confianza, prob_cracked


def crear_grafico_confianza(prob_cracked):
    """
    Crea un gráfico de barras con las probabilidades.
    
    Args:
        prob_cracked: Probabilidad de que sea fisura
    
    Returns:
        matplotlib.figure.Figure: Figura con el gráfico
    """
    prob_uncracked = 1.0 - prob_cracked
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    categorias = ['CON FISURA', 'SIN FISURA']
    probabilidades = [prob_cracked * 100, prob_uncracked * 100]
    colores = ['#e74c3c', '#2ecc71']
    
    barras = ax.barh(categorias, probabilidades, color=colores, alpha=0.8)
    
    # Añadir valores en las barras
    for i, (barra, prob) in enumerate(zip(barras, probabilidades)):
        ancho = barra.get_width()
        ax.text(ancho / 2, barra.get_y() + barra.get_height() / 2,
                f'{prob:.2f}%',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax.set_xlabel('Probabilidad (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_title('Confianza de la Predicción', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def interpretar_resultado(prob_cracked, umbral=0.5):
    """
    Genera una interpretación detallada del resultado.
    
    Args:
        prob_cracked: Probabilidad de fisura
        umbral: Umbral de decisión
    
    Returns:
        dict: Diccionario con interpretación y recomendaciones
    """
    resultado = {}
    
    if prob_cracked >= umbral:
        resultado['diagnostico'] = "⚠️ FISURA DETECTADA"
        resultado['color'] = "red"
        
        if prob_cracked >= 0.95:
            resultado['nivel'] = "ALTA"
            resultado['mensaje'] = "El modelo tiene muy alta confianza en la presencia de una fisura."
            resultado['recomendacion'] = "🔴 **Acción Urgente**: Se recomienda inspección profesional inmediata por un ingeniero estructural."
        elif prob_cracked >= 0.75:
            resultado['nivel'] = "MODERADA-ALTA"
            resultado['mensaje'] = "El modelo detecta una fisura con alta probabilidad."
            resultado['recomendacion'] = "🟠 **Atención Requerida**: Programar inspección profesional pronto."
        else:
            resultado['nivel'] = "MODERADA"
            resultado['mensaje'] = "El modelo sugiere la presencia de una posible fisura."
            resultado['recomendacion'] = "🟡 **Monitoreo**: Considerar inspección profesional y seguimiento fotográfico."
    else:
        resultado['diagnostico'] = "✅ SIN FISURA"
        resultado['color'] = "green"
        
        if prob_cracked <= 0.05:
            resultado['nivel'] = "CONFIANZA ALTA"
            resultado['mensaje'] = "El modelo tiene muy alta confianza en la ausencia de fisuras."
            resultado['recomendacion'] = "✅ **Estado Normal**: No se detectan patologías estructurales en esta imagen."
        elif prob_cracked <= 0.25:
            resultado['nivel'] = "CONFIANZA MODERADA"
            resultado['mensaje'] = "El modelo no detecta fisuras con buena confianza."
            resultado['recomendacion'] = "✅ **Estado Aceptable**: Continuar con inspecciones de rutina."
        else:
            resultado['nivel'] = "CONFIANZA BAJA"
            resultado['mensaje'] = "El modelo no está completamente seguro."
            resultado['recomendacion'] = "⚪ **Caso Límite**: Si hay dudas visuales, considerar segunda opinión."
    
    return resultado


def crear_overlay_segmentacion(imagen_original, mascara, opacidad=0.5):
    """
    Crea un overlay de la máscara de segmentación sobre la imagen original.
    
    Args:
        imagen_original: PIL Image
        mascara: numpy array (H, W) con valores 0-255
        opacidad: float, transparencia del overlay (0-1)
        
    Returns:
        PIL Image con overlay
    """
    # Convertir imagen original a numpy
    img_np = np.array(imagen_original.convert('RGB'))
    
    # Redimensionar máscara al tamaño original si es necesario
    if mascara.shape[:2] != img_np.shape[:2]:
        mascara = cv2.resize(mascara, (img_np.shape[1], img_np.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
    
    # Crear overlay rojo para fisuras
    overlay = img_np.copy()
    mascara_bool = (mascara > 127).astype(bool)
    
    # Colorear fisuras en rojo
    overlay[mascara_bool] = [255, 0, 0]  # Rojo brillante
    
    # Mezclar con imagen original
    resultado = cv2.addWeighted(img_np, 1 - opacidad, overlay, opacidad, 0)
    
    return Image.fromarray(resultado)


@st.cache_data(show_spinner=False)
def convertir_pil_a_bgr(imagen_hash: str, imagen_array: np.ndarray) -> np.ndarray:
    """Convierte imagen PIL a BGR con caché.
    
    Args:
        imagen_hash: Hash MD5 de la imagen (para caché)
        imagen_array: Array RGB de la imagen
    
    Returns:
        Array BGR para OpenCV
    """
    return cv2.cvtColor(imagen_array, cv2.COLOR_RGB2BGR)


@st.cache_data(show_spinner="⚙️ Calculando parámetros estructurales...")
def calcular_parametros_cacheados(mascara_bytes: bytes, imagen_hash: str, imagen_array: np.ndarray):
    """Calcula parámetros estructurales con caché para evitar recálculos.
    
    Args:
        mascara_bytes: Máscara serializada (para caché)
        imagen_hash: Hash de la imagen original
        imagen_array: Array RGB de la imagen
        
    Returns:
        Tupla con (ancho_dict, orientacion_dict, profundidad_dict)
    """
    # Deserializar máscara
    mascara = np.frombuffer(mascara_bytes, dtype=np.uint8).reshape(imagen_array.shape[:2])
    
    # Convertir imagen a BGR
    img_bgr = convertir_pil_a_bgr(imagen_hash, imagen_array)
    
    # Calcular parámetros (operaciones costosas)
    with st.spinner("📏 Midiendo ancho de fisura..."):
        ancho = medir_ancho_fisura(mascara, pixeles_por_mm=1.0)
    
    with st.spinner("🧭 Detectando orientación..."):
        orientacion = detectar_orientacion(mascara)
    
    with st.spinner("🔍 Estimando profundidad visual..."):
        profundidad = estimar_profundidad(img_bgr, mascara)
    
    return ancho, orientacion, profundidad


def mostrar_parametros_estructurales(mascara, imagen_original):
    """
    Calcula y muestra los parámetros estructurales de la fisura.
    
    Args:
        mascara: numpy array (H, W) con valores 0-255
        imagen_original: PIL Image
        
    Returns:
        dict con los parámetros calculados
    """
    try:
        # Convertir imagen a numpy para cálculos
        img_rgb = np.array(imagen_original.convert('RGB'))
        img_hash = calcular_hash_imagen(img_rgb)
        
        # Serializar máscara para caché (bytes son hashables)
        mascara_bytes = mascara.tobytes()
        
        # Calcular parámetros con caché
        ancho, orientacion, profundidad = calcular_parametros_cacheados(
            mascara_bytes, 
            img_hash, 
            img_rgb
        )
        
        # Mostrar parámetros SIEMPRE EXPANDIDOS
        st.markdown("### 📏 Ancho de Fisura")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ancho Promedio", f"{ancho.get('ancho_promedio_mm', 0):.2f} mm", help="Media del ancho en píxeles")
        col2.metric("Ancho Máximo", f"{ancho.get('ancho_maximo_mm', 0):.2f} mm", help="Apertura máxima detectada")
        col3.metric("Área Total", f"{ancho.get('area_total_mm2', 0):.2f} mm²", help="Superficie afectada")
        
        st.markdown("---")
        st.markdown("### 🧭 Orientación")
        col1, col2 = st.columns(2)
        col1.metric("Orientación", orientacion.get('orientacion', 'N/A'))
        col2.metric("Ángulo", f"{orientacion.get('angulo_grados', 0):.1f}°")
        
        confianza = orientacion.get('confianza', 0.0)
        st.progress(confianza, text=f"Confianza: {confianza*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### 🔍 Profundidad Visual")
        categoria = profundidad.get('profundidad_categoria', 'Desconocida')
        intensidad = profundidad.get('intensidad_promedio', 0)
        
        col1, col2 = st.columns(2)
        col1.metric("Categoría", categoria)
        col2.metric("Intensidad Media", f"{intensidad:.1f}")
        
        st.info(profundidad.get('advertencia', 'Estimación basada en análisis visual'))
        
        return {
            'ancho_promedio_mm': ancho.get('ancho_promedio_mm', 0),
            'ancho_maximo_mm': ancho.get('ancho_maximo_mm', 0),
            'area_total_mm2': ancho.get('area_total_mm2', 0),
            'orientacion': orientacion.get('orientacion', 'N/A'),
            'angulo_grados': orientacion.get('angulo_grados', 0),
            'profundidad_categoria': profundidad.get('profundidad_categoria', 'N/A')
        }
    except Exception as e:
        st.error(f"❌ Error al calcular parámetros: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación."""
    
    # ========================================================================
    # SIDEBAR - INFORMACIÓN Y CONFIGURACIÓN
    # ========================================================================
    
    st.sidebar.title("🏗️ Análisis de Fisuras")
    st.sidebar.markdown("---")
    
    # Selector de modo
    st.sidebar.header("🔧 Modo de Análisis")
    modo = st.sidebar.radio(
        "Selecciona el tipo de análisis:",
        ["🔍 Detección (Clasificación)", "📐 Segmentación (Parámetros)"],
        help="Detección: Clasifica si hay fisura o no.\nSegmentación: Analiza fisuras y mide parámetros estructurales."
    )
    
    # Determinar qué modo está seleccionado
    modo_deteccion = "Detección" in modo
    modo_segmentacion = "Segmentación" in modo
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Acerca de")
    
    if modo_deteccion:
        st.sidebar.info(
            """
            **Modo: Detección**
            
            Utiliza **MobileNetV2** para clasificar imágenes 
            como "Con Fisura" o "Sin Fisura".
            
            **Métricas:**
            - Precisión: 94.36%
            - Recall: 99.64%
            - F1-Score: 96.77%
            
            **Dataset:**
            SDNET2018 (56,092 imágenes)
            """
        )
    else:
        st.sidebar.info(
            """
            **Modo: Segmentación**
            
            Utiliza **U-Net Lite** para segmentar fisuras 
            y medir parámetros estructurales.
            
            **Métricas:**
            - IoU: 60.5%
            - Dice: 73.0%
            - Accuracy: 97.4%
            
            **Parámetros medidos:**
            - Ancho de fisura (mm)
            - Orientación (H/V/D)
            - Profundidad visual
            
            **Dataset:**
            CRACK500 (3,368 pares)
            """
        )
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Configuración")
    
    if modo_deteccion:
        umbral = st.sidebar.slider(
            "Umbral de Decisión",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Umbral para clasificar como fisura. Valores más bajos detectan más fisuras (más sensible)."
        )
    else:
        opacidad_overlay = st.sidebar.slider(
            "Opacidad del Overlay",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Transparencia de la máscara sobre la imagen original."
        )
        umbral = 0.5  # Valor por defecto para segmentación
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Estado de Modelos")
    
    # Cargar modelo(s)
    modelo_det, nombre_modelo_det = cargar_modelo()
    st.sidebar.success(f"✅ Detección: `{nombre_modelo_det}`")
    
    if modo_segmentacion:
        if SEGMENTACION_DISPONIBLE:
            modelo_seg = cargar_modelo_segmentacion()
            if modelo_seg:
                st.sidebar.success("✅ Segmentación: `unet_segmentacion_final.keras`")
            else:
                st.sidebar.error("❌ Modelo de segmentación no disponible")
                st.error("❌ No se pudo cargar el modelo de segmentación. Usa modo Detección.")
                return
        else:
            st.sidebar.error("❌ Módulo de segmentación no disponible")
            st.error("❌ El módulo de segmentación no está instalado. Usa modo Detección.")
            return
    
    if modo_deteccion:
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Precisión", "94.36%")
        col2.metric("Recall", "99.64%")
        
        col3, col4 = st.sidebar.columns(2)
        col3.metric("F1-Score", "96.77%")
        col4.metric("AUC", "94.13%")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 - Sistema de Análisis de Fisuras")
    
    # ========================================================================
    # ÁREA PRINCIPAL
    # ========================================================================
    
    if modo_deteccion:
        st.title("🔍 Sistema de Detección de Fisuras en Estructuras")
        st.markdown(
            """
            Sube una fotografía de una estructura de concreto y el sistema analizará 
            si presenta **fisuras** o si está en **buen estado**.
            """
        )
    else:
        st.title("📐 Sistema de Análisis de Parámetros Estructurales")
        st.markdown(
            """
            Sube una fotografía de una fisura y el sistema generará una **segmentación detallada** 
            con mediciones de **ancho**, **orientación** y **profundidad visual**.
            """
        )

    
    st.markdown("---")
    
    # Subida de archivo
    uploaded_file = st.file_uploader(
        "📁 Selecciona una imagen",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Cargar imagen
        try:
            imagen_original = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error al cargar la imagen: {e}")
            return
        
        # ========== MODO DETECCIÓN ==========
        if modo_deteccion:
            # Layout de dos columnas
            col_izq, col_der = st.columns(2)
            
            with col_izq:
                st.subheader("📷 Imagen Original")
                st.image(imagen_original, use_container_width=True, caption=f"Tamaño: {imagen_original.size[0]}x{imagen_original.size[1]} px")
            
            with col_der:
                st.subheader("🤖 Análisis del Modelo")
                
                with st.spinner("Analizando imagen..."):
                    # Preprocesar
                    img_prep = preprocesar_imagen(imagen_original)
                    
                    # Predecir
                    clase, confianza, prob_cracked = predecir(modelo_det, img_prep, umbral)
                    
                    # Interpretar
                    interpretacion = interpretar_resultado(prob_cracked, umbral)
                
                # Mostrar resultado principal
                if interpretacion['color'] == 'red':
                    st.error(f"### {interpretacion['diagnostico']}")
                else:
                    st.success(f"### {interpretacion['diagnostico']}")
                
                st.markdown(f"**Confianza:** {confianza * 100:.2f}%")
                st.markdown(f"**Nivel de Confianza:** {interpretacion['nivel']}")
            
            # Gráfico de confianza
            st.markdown("---")
            st.subheader("📊 Distribución de Probabilidades")
            
            fig = crear_grafico_confianza(prob_cracked)
            st.pyplot(fig)
            plt.close(fig)
            
            # Interpretación detallada
            st.markdown("---")
            st.subheader("📋 Interpretación Detallada")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.info(f"**Análisis:**\n\n{interpretacion['mensaje']}")
            
            with col_info2:
                if interpretacion['color'] == 'red':
                    st.warning(f"**Recomendación:**\n\n{interpretacion['recomendacion']}")
                else:
                    st.success(f"**Recomendación:**\n\n{interpretacion['recomendacion']}")
            
            # Detalles técnicos (expandible)
            with st.expander("🔬 Ver Detalles Técnicos"):
                img_size_display = f"{IMG_SIZE}x{IMG_SIZE}" if isinstance(IMG_SIZE, int) else f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
                st.markdown(f"""
                **Probabilidades:**
                - Probabilidad de FISURA: {prob_cracked * 100:.4f}%
                - Probabilidad de SIN FISURA: {(1 - prob_cracked) * 100:.4f}%
                
                **Configuración:**
                - Umbral de decisión: {umbral}
                - Tamaño de entrada: {img_size_display} px
                - Modelo: {nombre_modelo_det}
                
                **Preprocesamiento:**
                - Redimensionado: {imagen_original.size} → {img_size_display}
                - Normalización: [0, 255] → [0, 1]
                - Modo de color: {imagen_original.mode} → RGB
                """)
            
            # Advertencia legal
            st.markdown("---")
            st.warning(
                """
                ⚠️ **Nota Importante:** Este sistema es una herramienta de apoyo y no reemplaza 
                la inspección profesional de un ingeniero estructural certificado. 
                Siempre consulte con un profesional para decisiones críticas sobre seguridad estructural.
                """
            )

        
        # ========== MODO SEGMENTACIÓN ==========
        else:
            # Generar segmentación
            with st.spinner("🔄 Generando segmentación y midiendo parámetros..."):
                img_np = np.array(imagen_original.convert('RGB'))
                mascara = modelo_seg.predecir(img_np, umbral=0.5)
                imagen_overlay = crear_overlay_segmentacion(imagen_original, mascara, opacidad_overlay)
            
            # Layout de dos columnas
            col_izq, col_der = st.columns(2)
            
            with col_izq:
                st.subheader("📷 Imagen Original")
                st.image(imagen_original, use_container_width=True, 
                         caption=f"Tamaño: {imagen_original.size[0]}x{imagen_original.size[1]} px")
                
                st.subheader("🎨 Segmentación de Fisuras")
                st.image(imagen_overlay, use_container_width=True, 
                         caption="Fisuras detectadas en rojo")
            
            with col_der:
                st.subheader("📏 Parámetros Estructurales")
                
                # Lazy loading: solo calcular parámetros si usuario lo solicita
                calcular_params = st.checkbox(
                    "🔬 Calcular Mediciones Detalladas",
                    value=True,
                    help="Desactiva para vista rápida. Activa para medir ancho, orientación y profundidad (10-20s en primera ejecución, luego instantáneo por caché)"
                )
                
                parametros = None
                if calcular_params:
                    parametros = mostrar_parametros_estructurales(mascara, imagen_original)
                else:
                    st.info("ℹ️ Marca la casilla para calcular parámetros estructurales detallados")
            
            # Detalles técnicos de la segmentación
            st.markdown("---")
            with st.expander("🔬 Ver Detalles Técnicos de Segmentación"):
                pixels_fisura = np.sum(mascara > 0)
                pixels_total = mascara.size
                porcentaje_fisura = (pixels_fisura / pixels_total) * 100
                
                st.markdown(f"""
                **Estadísticas de Segmentación:**
                - Píxeles de fisura detectados: {pixels_fisura:,}
                - Píxeles totales: {pixels_total:,}
                - Porcentaje de área con fisura: {porcentaje_fisura:.2f}%
                
                **Configuración:**
                - Umbral de segmentación: {umbral}
                - Opacidad del overlay: {opacidad_overlay}
                - Modelo: U-Net Lite (1.95M parámetros)
                - Tamaño de entrada: 128x128 px
                
                **Preprocesamiento:**
                - Redimensionado: {imagen_original.size} → 128x128
                - Normalización: [0, 255] → [0, 1]
                - Modo de color: {imagen_original.mode} → RGB
                """)
                
                if parametros:
                    st.markdown(f"""
                    **Parámetros Medidos:**
                    - Ancho promedio: {parametros.get('ancho_promedio_mm', 0):.2f} mm
                    - Ancho máximo: {parametros.get('ancho_maximo_mm', 0):.2f} mm
                    - Área total: {parametros.get('area_total_mm2', 0):.2f} mm²
                    - Orientación: {parametros.get('orientacion', 'N/A')}
                    - Profundidad visual: {parametros.get('profundidad_categoria', 'N/A')}
                    """)
            
            # Advertencia legal
            st.markdown("---")
            st.warning(
                """
                ⚠️ **Nota Importante:** Este sistema es una herramienta de apoyo y no reemplaza 
                la inspección profesional de un ingeniero estructural certificado. 
                Las mediciones son estimaciones visuales y deben ser validadas por profesionales.
                """
            )

    
    else:
        # Mensaje cuando no hay imagen
        st.info("👆 Por favor, sube una imagen para comenzar el análisis.")
        
        # Ejemplos de uso
        st.markdown("---")
        st.subheader("💡 Consejos para Mejores Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                **📸 Calidad de Imagen**
                - Usa buena iluminación
                - Evita sombras fuertes
                - Enfoca la zona de interés
                """
            )
        
        with col2:
            st.markdown(
                """
                **📏 Encuadre**
                - Captura perpendicular
                - Distancia adecuada (0.5-2m)
                - Centra la fisura sospechosa
                """
            )
        
        with col3:
            st.markdown(
                """
                **✅ Formato**
                - JPG, JPEG o PNG
                - Resolución mínima: 224x224
                - Máximo recomendado: 4096x4096
                """
            )


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
