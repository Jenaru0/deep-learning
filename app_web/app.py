"""
Aplicación Web para Detección de Fisuras en Estructuras
========================================================

Interfaz gráfica desarrollada con Streamlit para detectar fisuras
en imágenes de estructuras utilizando el modelo entrenado MobileNetV2.

Autor: Sistema de Detección de Fisuras
Fecha: Octubre 2025
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import sys
from pathlib import Path

# Configurar el path para importar módulos del proyecto
PROYECTO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROYECTO_ROOT))

try:
    from config import RUTA_MODELO_DETECCION as MODELOS_DIR, IMG_SIZE
except ImportError:
    # Fallback si config.py no existe o está mal configurado
    MODELOS_DIR = str(PROYECTO_ROOT / "modelos" / "deteccion")
    IMG_SIZE = 224


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

@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo entrenado.
    Usa cache de Streamlit para cargar solo una vez.
    
    Returns:
        tensorflow.keras.Model: Modelo cargado
    """
    modelos_dir = Path(MODELOS_DIR)
    
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
        st.info("Por favor, entrena un modelo primero ejecutando: `python3 scripts/entrenamiento/entrenar_deteccion.py`")
        st.stop()
    
    try:
        modelo = tf.keras.models.load_model(modelo_path)
        return modelo, modelo_path.name
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()


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


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    """Función principal de la aplicación."""
    
    # ========================================================================
    # SIDEBAR - INFORMACIÓN Y CONFIGURACIÓN
    # ========================================================================
    
    st.sidebar.title("🏗️ Detector de Fisuras")
    st.sidebar.markdown("---")
    
    st.sidebar.header("ℹ️ Acerca de")
    st.sidebar.info(
        """
        Esta aplicación utiliza **Deep Learning** (MobileNetV2) 
        para detectar fisuras en estructuras de concreto.
        
        **Características:**
        - Precisión: 94.36%
        - Recall: 99.64%
        - F1-Score: 96.77%
        
        **Dataset de entrenamiento:**
        SDNET2018 (56,092 imágenes)
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Configuración")
    
    umbral = st.sidebar.slider(
        "Umbral de Decisión",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Umbral para clasificar como fisura. Valores más bajos detectan más fisuras (más sensible)."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Métricas del Modelo")
    
    # Cargar modelo
    modelo, nombre_modelo = cargar_modelo()
    
    st.sidebar.success(f"✅ Modelo cargado: `{nombre_modelo}`")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Precisión", "94.36%")
    col2.metric("Recall", "99.64%")
    
    col3, col4 = st.sidebar.columns(2)
    col3.metric("F1-Score", "96.77%")
    col4.metric("AUC", "94.13%")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 - Sistema de Detección de Fisuras")
    
    # ========================================================================
    # ÁREA PRINCIPAL
    # ========================================================================
    
    st.title("🔍 Sistema de Detección de Fisuras en Estructuras")
    st.markdown(
        """
        Sube una fotografía de una estructura de concreto y el sistema analizará 
        si presenta **fisuras** o si está en **buen estado**.
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
        
        # Layout de dos columnas
        col_izq, col_der = st.columns(2)
        
        with col_izq:
            st.subheader("📷 Imagen Original")
            st.image(imagen_original, use_column_width=True, caption=f"Tamaño: {imagen_original.size[0]}x{imagen_original.size[1]} px")
        
        with col_der:
            st.subheader("🤖 Análisis del Modelo")
            
            with st.spinner("Analizando imagen..."):
                # Preprocesar
                img_prep = preprocesar_imagen(imagen_original)
                
                # Predecir
                clase, confianza, prob_cracked = predecir(modelo, img_prep, umbral)
                
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
            - Modelo: {nombre_modelo}
            
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
