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
except Exception as e:
    # Si falla la descarga, continuar (los modelos pueden estar localmente)
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
    # HEADER PRINCIPAL CON DISEÑO MODERNO
    # ========================================================================
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; 
                    border-radius: 1rem; 
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>
                🏗️ Sistema de Análisis de Fisuras
            </h1>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                🤖 Inteligencia Artificial para Inspección Estructural
            </p>
            <div style='background: rgba(255,255,255,0.2); height: 2px; margin: 1rem 0 0.5rem 0; border-radius: 2px;'></div>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.95rem;'>
                Análisis automático de fisuras en concreto utilizando Deep Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ========================================================================
    # TABS PRINCIPALES (En vez de radio buttons en sidebar)
    # ========================================================================
    tab_deteccion, tab_segmentacion, tab_ayuda = st.tabs([
        "🔍 Detección Rápida", 
        "📐 Análisis Detallado",
        "❓ Ayuda"
    ])
    
    # ========================================================================
    # TAB 1: DETECCIÓN
    # ========================================================================
    with tab_deteccion:
        modo_deteccion = True
        modo_segmentacion = False
        
        # Descripción del modo con diseño moderno
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); 
                        padding: 1.5rem; 
                        border-left: 5px solid #00acc1; 
                        border-radius: 0.75rem; 
                        margin-bottom: 1.5rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                <h3 style='margin: 0 0 0.75rem 0; color: #00695c; font-weight: 600;'>
                    🔍 Detección Rápida - Clasificación Binaria
                </h3>
                <p style='margin: 0 0 0.5rem 0; color: #004d40; line-height: 1.6;'>
                    Utiliza <strong>MobileNetV2</strong> entrenado con 56,092 imágenes del dataset SDNET2018.
                </p>
                <div style='background: white; padding: 0.75rem; border-radius: 0.5rem; margin-top: 0.75rem;'>
                    <span style='color: #00897b; font-weight: 600;'>📊 Métricas:</span>
                    <span style='color: #00695c;'>
                        Precisión <strong>94.36%</strong> | 
                        Recall <strong>99.64%</strong> | 
                        F1-Score <strong>96.77%</strong>
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Configuración en columnas compactas
        col_conf1, col_conf2 = st.columns([3, 1])
        with col_conf1:
            uploaded_file_det = st.file_uploader(
                "📁 Sube una imagen de la estructura",
                type=['jpg', 'jpeg', 'png'],
                help="Formatos: JPG, JPEG, PNG | Resolución recomendada: 224x224 o superior",
                key="upload_deteccion"
            )
        with col_conf2:
            umbral = st.slider(
                "Umbral",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Sensibilidad de detección"
            )
        
        if uploaded_file_det is not None:
            # Ocultar instrucciones iniciales (UX: reducir ruido visual)
            
            # Cargar modelo SOLO cuando hay imagen (lazy loading)
            with st.spinner("⏳ Cargando modelo MobileNetV2..."):
                modelo_det, nombre_modelo_det = cargar_modelo()
            
            # Cargar imagen
            try:
                imagen_original = Image.open(uploaded_file_det)
            except Exception as e:
                st.error(f"❌ Error al cargar la imagen: {e}")
                return
            
            # Progress bar para feedback visual
            progress_bar = st.progress(0, text="🔄 Procesando imagen...")
            
            # Layout de dos columnas
            col_izq, col_der = st.columns([1, 1])
            
            with col_izq:
                st.markdown("#### 📷 Imagen Original")
                st.image(
                    imagen_original, 
                    use_container_width=True, 
                    caption=f"{imagen_original.size[0]}×{imagen_original.size[1]} px"
                )
            
            progress_bar.progress(30, text="🧠 Ejecutando modelo...")
            
            # Preprocesar y predecir
            img_prep = preprocesar_imagen(imagen_original)
            clase, confianza, prob_cracked = predecir(modelo_det, img_prep, umbral)
            interpretacion = interpretar_resultado(prob_cracked, umbral)
            
            progress_bar.progress(70, text="📊 Generando resultados...")
            
            with col_der:
                st.markdown("#### 🤖 Resultado del Análisis")
                
                # Card de resultado principal con color dinámico
                if interpretacion['color'] == 'red':
                    resultado_bg = "#fee"
                    resultado_border = "#f44"
                    resultado_icon = "⚠️"
                else:
                    resultado_bg = "#efe"
                    resultado_border = "#4a4"
                    resultado_icon = "✅"
                
                st.markdown(
                    f"""
                    <div style='background-color: {resultado_bg}; padding: 1.5rem; border-left: 5px solid {resultado_border}; border-radius: 0.5rem; margin-bottom: 1rem;'>
                    <h2 style='margin: 0 0 0.5rem 0; color: {resultado_border};'>{resultado_icon} {interpretacion['diagnostico']}</h2>
                    <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
                    <strong>Confianza:</strong> <span style='font-size: 1.5rem; font-weight: bold;'>{confianza * 100:.1f}%</span>
                    </div>
                    <div style='background-color: white; padding: 0.5rem; border-radius: 0.3rem; margin-top: 0.5rem;'>
                    <strong>Nivel:</strong> {interpretacion['nivel']}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Mensaje de interpretación
                st.info(f"💡 **{interpretacion['mensaje']}**")
                
                # Recomendación con color apropiado
                if interpretacion['color'] == 'red':
                    st.warning(f"**📋 Recomendación:**\n\n{interpretacion['recomendacion']}")
                else:
                    st.success(f"**📋 Recomendación:**\n\n{interpretacion['recomendacion']}")
            
            progress_bar.progress(100, text="✅ Análisis completado")
            # Ocultar progress bar después de completar (UX limpia)
            import time
            time.sleep(0.5)
            progress_bar.empty()
            
            # Gráfico de confianza
            st.markdown("---")
            st.markdown("#### 📊 Distribución de Probabilidades")
            fig = crear_grafico_confianza(prob_cracked)
            st.pyplot(fig)
            plt.close(fig)
            
            # Detalles técnicos (colapsados por defecto - UX: mostrar solo lo esencial)
            with st.expander("🔬 Detalles Técnicos", expanded=False):
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("Probabilidad Fisura", f"{prob_cracked * 100:.2f}%")
                    st.metric("Umbral Decisión", f"{umbral}")
                    st.metric("Tamaño Entrada", f"{IMG_SIZE}×{IMG_SIZE} px")
                with col_t2:
                    st.metric("Probabilidad Sin Fisura", f"{(1 - prob_cracked) * 100:.2f}%")
                    st.metric("Modelo", nombre_modelo_det)
                    st.metric("Resolución Original", f"{imagen_original.size[0]}×{imagen_original.size[1]} px")
            
            # Advertencia legal (siempre visible pero discreta)
            st.markdown("---")
            st.caption(
                "⚠️ **Disclaimer:** Este sistema es una herramienta de apoyo. "
                "No reemplaza la inspección de un ingeniero estructural certificado."
            )
        
        else:
            # Instrucciones cuando no hay imagen
            st.markdown(
                """
                <h3 style='color: #424242; margin-bottom: 1.5rem;'>👆 Sube una imagen para comenzar</h3>
                """,
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    """
                    <div style='background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                                padding: 1.25rem; 
                                border-radius: 0.75rem; 
                                border-left: 4px solid #f57c00;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                                height: 100%;'>
                        <h4 style='margin: 0 0 0.75rem 0; color: #e65100;'>📸 Calidad</h4>
                        <ul style='margin: 0; padding-left: 1.25rem; color: #bf360c;'>
                            <li>✓ Buena iluminación</li>
                            <li>✓ Sin sombras fuertes</li>
                            <li>✓ Imagen enfocada</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    """
                    <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                                padding: 1.25rem; 
                                border-radius: 0.75rem; 
                                border-left: 4px solid #388e3c;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                                height: 100%;'>
                        <h4 style='margin: 0 0 0.75rem 0; color: #1b5e20;'>📏 Encuadre</h4>
                        <ul style='margin: 0; padding-left: 1.25rem; color: #2e7d32;'>
                            <li>✓ Toma perpendicular</li>
                            <li>✓ Distancia 0.5-2m</li>
                            <li>✓ Fisura centrada</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    """
                    <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                                padding: 1.25rem; 
                                border-radius: 0.75rem; 
                                border-left: 4px solid #1976d2;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                                height: 100%;'>
                        <h4 style='margin: 0 0 0.75rem 0; color: #0d47a1;'>✅ Formato</h4>
                        <ul style='margin: 0; padding-left: 1.25rem; color: #1565c0;'>
                            <li>✓ JPG, JPEG, PNG</li>
                            <li>✓ Mín: 224×224 px</li>
                            <li>✓ Máx: 4096×4096 px</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # ========================================================================
    # TAB 2: SEGMENTACIÓN
    # ========================================================================
    with tab_segmentacion:
        modo_deteccion = False
        modo_segmentacion = True
        
        # Verificar disponibilidad
        if not SEGMENTACION_DISPONIBLE:
            st.error("❌ **Módulo de segmentación no disponible**")
            st.info("💡 Usa la pestaña **Detección Rápida** para análisis de fisuras.")
            return
        
        # Descripción del modo con diseño moderno
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%); 
                        padding: 1.5rem; 
                        border-left: 5px solid #d81b60; 
                        border-radius: 0.75rem; 
                        margin-bottom: 1.5rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                <h3 style='margin: 0 0 0.75rem 0; color: #880e4f; font-weight: 600;'>
                    📐 Análisis Detallado - Segmentación Semántica
                </h3>
                <p style='margin: 0 0 0.5rem 0; color: #6a1b32; line-height: 1.6;'>
                    Utiliza <strong>U-Net Lite</strong> entrenado con 3,368 pares del dataset CRACK500.
                </p>
                <div style='background: white; padding: 0.75rem; border-radius: 0.5rem; margin-top: 0.75rem;'>
                    <span style='color: #c2185b; font-weight: 600;'>📊 Métricas:</span>
                    <span style='color: #880e4f;'>
                        IoU <strong>60.5%</strong> | 
                        Dice <strong>73.0%</strong> | 
                        Accuracy <strong>97.4%</strong>
                    </span>
                    <br>
                    <span style='color: #c2185b; font-weight: 600; margin-top: 0.5rem; display: inline-block;'>📏 Mediciones:</span>
                    <span style='color: #880e4f;'>
                        Ancho | Orientación | Profundidad Visual
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Configuración
        col_conf1, col_conf2 = st.columns([3, 1])
        with col_conf1:
            uploaded_file_seg = st.file_uploader(
                "📁 Sube una imagen con fisura visible",
                type=['jpg', 'jpeg', 'png'],
                help="Para mejores resultados, captura cercana de la fisura",
                key="upload_segmentacion"
            )
        with col_conf2:
            opacidad_overlay = st.slider(
                "Opacidad",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Transparencia del overlay"
            )
        
        if uploaded_file_seg is not None:
            # Cargar modelos
            with st.spinner("⏳ Cargando modelos..."):
                modelo_det, _ = cargar_modelo()
                modelo_seg = cargar_modelo_segmentacion()
                
            if modelo_seg is None:
                st.error("❌ No se pudo cargar el modelo de segmentación")
                return
            
            # Cargar imagen
            try:
                imagen_original = Image.open(uploaded_file_seg)
            except Exception as e:
                st.error(f"❌ Error al cargar la imagen: {e}")
                return
            
            # Progress tracking
            progress_bar = st.progress(0, text="🔄 Procesando imagen...")
            
            progress_bar.progress(20, text="🧠 Ejecutando segmentación...")
            
            # Generar segmentación
            img_np = np.array(imagen_original.convert('RGB'))
            mascara = modelo_seg.predecir(img_np, umbral=0.5)
            
            progress_bar.progress(50, text="🎨 Generando visualización...")
            imagen_overlay = crear_overlay_segmentacion(imagen_original, mascara, opacidad_overlay)
            
            progress_bar.progress(70, text="📏 Calculando estadísticas...")
            
            # Calcular estadísticas básicas
            pixels_fisura = np.sum(mascara > 0)
            pixels_total = mascara.size
            porcentaje_fisura = (pixels_fisura / pixels_total) * 100
            
            progress_bar.progress(100, text="✅ Segmentación completada")
            import time
            time.sleep(0.5)
            progress_bar.empty()
            
            # Layout de dos columnas
            col_izq, col_der = st.columns([1, 1])
            
            with col_izq:
                st.markdown("#### 📷 Imagen Original")
                st.image(
                    imagen_original, 
                    use_container_width=True,
                    caption=f"{imagen_original.size[0]}×{imagen_original.size[1]} px"
                )
                
                # Estadísticas rápidas
                st.markdown(
                    f"""
                    <div style='background-color: #f8f9fa; padding: 0.8rem; border-radius: 0.3rem; margin-top: 1rem;'>
                    <strong>📊 Cobertura de Fisura:</strong> {porcentaje_fisura:.2f}%<br>
                    <strong>🔢 Píxeles Detectados:</strong> {pixels_fisura:,} / {pixels_total:,}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_der:
                st.markdown("#### 🎨 Segmentación")
                st.image(
                    imagen_overlay,
                    use_container_width=True,
                    caption="Fisuras en rojo"
                )
                
                # Toggle para cálculos pesados (UX: dar control al usuario)
                calcular_params = st.toggle(
                    "🔬 Calcular Parámetros Estructurales",
                    value=False,
                    help="Mediciones detalladas (10-20s en primera ejecución, luego instantáneo)"
                )
            
            # Parámetros estructurales (solo si se solicita)
            if calcular_params:
                st.markdown("---")
                st.markdown("### 📏 Parámetros Estructurales Detallados")
                
                with st.spinner("⚙️ Calculando mediciones..."):
                    parametros = mostrar_parametros_estructurales(mascara, imagen_original)
                
                if parametros:
                    # Resumen en cards
                    st.markdown("#### 📊 Resumen de Mediciones")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Ancho Promedio", f"{parametros['ancho_promedio_mm']:.2f} mm")
                    col2.metric("Ancho Máximo", f"{parametros['ancho_maximo_mm']:.2f} mm")
                    col3.metric("Área Total", f"{parametros['area_total_mm2']:.2f} mm²")
                    col4.metric("Orientación", parametros['orientacion'])
            
            # Detalles técnicos (colapsados)
            with st.expander("🔬 Detalles Técnicos de Segmentación", expanded=False):
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.metric("Píxeles de Fisura", f"{pixels_fisura:,}")
                    st.metric("Porcentaje Área", f"{porcentaje_fisura:.2f}%")
                    st.metric("Umbral", "0.5")
                with col_t2:
                    st.metric("Píxeles Totales", f"{pixels_total:,}")
                    st.metric("Opacidad Overlay", f"{opacidad_overlay}")
                    st.metric("Modelo", "U-Net Lite (1.95M)")
            
            st.markdown("---")
            st.caption(
                "⚠️ **Disclaimer:** Las mediciones son estimaciones visuales. "
                "Validar con profesionales e instrumentos calibrados."
            )
        
        else:
            # Instrucciones
            st.markdown(
                """
                <h3 style='color: #424242; margin-bottom: 1.5rem;'>👆 Sube una imagen para análisis detallado</h3>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <div style='background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%); 
                            padding: 1.5rem; 
                            border-left: 4px solid #3f51b5; 
                            border-radius: 0.75rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                    <h4 style='margin: 0 0 0.75rem 0; color: #1a237e;'>💡 Consejos para Segmentación</h4>
                    <ul style='margin: 0; padding-left: 1.5rem; color: #283593; line-height: 1.8;'>
                        <li>Captura cercana de la fisura (distancia 0.3-1m)</li>
                        <li>Asegura que la fisura sea claramente visible</li>
                        <li>Evita reflejos y sombras sobre la fisura</li>
                        <li>Preferible fondo de concreto uniforme</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # ========================================================================
    # TAB 3: AYUDA
    # ========================================================================
    with tab_ayuda:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%); 
                        padding: 1.5rem; 
                        border-left: 5px solid #f9a825; 
                        border-radius: 0.75rem;
                        margin-bottom: 2rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
                <h2 style='margin: 0 0 0.5rem 0; color: #f57f17;'>📚 Guía de Uso</h2>
                <p style='margin: 0; color: #f9a825;'>
                    Todo lo que necesitas saber para aprovechar al máximo el sistema
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Sección de FAQ
        st.markdown(
            """
            <h3 style='color: #424242; margin-bottom: 1rem;'>❓ Preguntas Frecuentes</h3>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("¿Cuál es la diferencia entre Detección y Segmentación?"):
            st.markdown(
                """
                **🔍 Detección Rápida:**
                - Clasifica la imagen completa: "Con Fisura" o "Sin Fisura"
                - Rápido (1-2 segundos)
                - Ideal para inspecciones masivas
                - Precisión: 94.36%
                
                **📐 Análisis Detallado:**
                - Identifica píxel por píxel dónde están las fisuras
                - Mide ancho, orientación y profundidad
                - Más lento (10-30 segundos con mediciones)
                - Ideal para análisis estructural específico
                """
            )
        
        with st.expander("¿Qué tipo de imágenes funcionan mejor?"):
            st.markdown(
                """
                **✅ Mejores resultados:**
                - Iluminación natural y uniforme
                - Toma perpendicular a la superficie
                - Fisura claramente visible
                - Fondo de concreto uniforme
                - Resolución mínima 224×224 px
                
                **❌ Evitar:**
                - Sombras fuertes o reflejos
                - Ángulos muy oblicuos
                - Imágenes borrosas o con mucho ruido
                - Resoluciones muy bajas (<200 px)
                """
            )
        
        with st.expander("¿Qué significan las métricas del modelo?"):
            st.markdown(
                """
                **Precisión (94.36%):** De todas las predicciones "Con Fisura", el 94.36% son correctas.
                
                **Recall (99.64%):** De todas las fisuras reales, el 99.64% son detectadas (muy pocas se pierden).
                
                **F1-Score (96.77%):** Balance entre Precisión y Recall.
                
                **IoU (60.5%):** Intersection over Union - solapamiento entre fisura predicha y real.
                """
            )
        
        with st.expander("¿Puedo confiar 100% en los resultados?"):
            st.markdown(
                """
                ⚠️ **No.** Este sistema es una **herramienta de apoyo**, no un reemplazo de ingenieros.
                
                **Úsalo para:**
                - Inspecciones preliminares
                - Priorizar estructuras que requieren atención
                - Documentación fotográfica
                
                **NO lo uses como:**
                - Única base para decisiones críticas de seguridad
                - Reemplazo de inspección profesional
                - Certificación estructural oficial
                
                ✅ **Siempre consulta con un ingeniero estructural certificado.**
                """
            )
        
        st.markdown(
            """
            <div style='height: 2px; background: linear-gradient(90deg, transparent, #9e9e9e, transparent); margin: 2rem 0;'></div>
            <h3 style='color: #424242; margin-bottom: 1.5rem;'>📊 Información Técnica</h3>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div style='background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%); 
                            padding: 1.5rem; 
                            border-left: 4px solid #0288d1; 
                            border-radius: 0.75rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                            height: 100%;'>
                    <h4 style='margin: 0 0 1rem 0; color: #01579b;'>🔍 Modelo de Detección</h4>
                    <ul style='margin: 0; padding-left: 1.5rem; color: #0277bd; line-height: 1.8;'>
                        <li><strong>Arquitectura:</strong> MobileNetV2</li>
                        <li><strong>Dataset:</strong> SDNET2018 (56,092 imgs)</li>
                        <li><strong>Entrenamiento:</strong> Transfer Learning</li>
                        <li><strong>Entrada:</strong> 224×224 RGB</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """
                <div style='background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                            padding: 1.5rem; 
                            border-left: 4px solid #8e24aa; 
                            border-radius: 0.75rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                            height: 100%;'>
                    <h4 style='margin: 0 0 1rem 0; color: #4a148c;'>📐 Modelo de Segmentación</h4>
                    <ul style='margin: 0; padding-left: 1.5rem; color: #6a1b9a; line-height: 1.8;'>
                        <li><strong>Arquitectura:</strong> U-Net Lite</li>
                        <li><strong>Dataset:</strong> CRACK500 (3,368 pares)</li>
                        <li><strong>Parámetros:</strong> 1.95M</li>
                        <li><strong>Entrada:</strong> 128×128 RGB</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown(
            """
            <div style='height: 2px; background: linear-gradient(90deg, transparent, #9e9e9e, transparent); margin: 2rem 0;'></div>
            <div style='background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%); 
                        padding: 1rem; 
                        border-radius: 0.5rem; 
                        text-align: center;
                        margin-top: 1rem;'>
                <p style='margin: 0; color: #616161; font-size: 0.9rem;'>
                    © 2025 - Sistema de Análisis de Fisuras | Desarrollado con ❤️ usando Streamlit + TensorFlow
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
