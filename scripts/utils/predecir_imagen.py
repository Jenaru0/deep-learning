"""
Script de Predicción: Detectar Fisuras en Imagen Individual
============================================================

Uso:
    python3 scripts/utils/predecir_imagen.py --imagen ruta/a/imagen.jpg
    
Ejemplos:
    python3 scripts/utils/predecir_imagen.py --imagen foto_pared.jpg
    python3 scripts/utils/predecir_imagen.py --imagen C:/Users/jonna/Desktop/test.jpg --umbral 0.5

Autor: Jesus Naranjo
Fecha: Octubre 2025
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importar TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Agregar ruta del proyecto
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RUTA_MODELO_DETECCION, IMG_SIZE

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODELO_PATH = Path(RUTA_MODELO_DETECCION) / "modelo_deteccion_final.keras"
UMBRAL_DEFAULT = 0.5  # Umbral de clasificación (ajustable)

# ============================================================================
# FUNCIONES
# ============================================================================

def cargar_modelo_deteccion():
    """Carga el modelo entrenado"""
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en: {MODELO_PATH}\n"
            "Asegúrate de haber entrenado el modelo primero."
        )
    
    print(f"📂 Cargando modelo desde: {MODELO_PATH}")
    modelo = load_model(MODELO_PATH)
    print(f"✅ Modelo cargado exitosamente\n")
    return modelo


def preprocesar_imagen(ruta_imagen, mostrar=False):
    """
    Preprocesa una imagen para el modelo.
    
    Args:
        ruta_imagen (str): Ruta a la imagen
        mostrar (bool): Si mostrar la imagen original
        
    Returns:
        np.array: Imagen preprocesada (1, 224, 224, 3)
    """
    # Cargar imagen
    img = image.load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Mostrar imagen si se solicita
    if mostrar:
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Imagen Original: {Path(ruta_imagen).name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Convertir a array y normalizar
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizar [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Añadir batch dimension
    
    return img_array


def predecir_fisura(modelo, img_array, umbral=0.5):
    """
    Predice si hay fisura en la imagen.
    
    Args:
        modelo: Modelo de Keras
        img_array: Imagen preprocesada
        umbral: Umbral de clasificación (0-1)
        
    Returns:
        dict: Resultado de predicción
    """
    # Realizar predicción
    probabilidad = modelo.predict(img_array, verbose=0)[0][0]
    
    # Clasificar según umbral
    # Modelo entrenado: 0 = uncracked, 1 = cracked
    tiene_fisura = probabilidad >= umbral
    
    resultado = {
        'probabilidad_fisura': float(probabilidad),
        'probabilidad_sin_fisura': float(1 - probabilidad),
        'tiene_fisura': tiene_fisura,
        'clase': 'CRACKED' if tiene_fisura else 'UNCRACKED',
        'confianza': float(probabilidad if tiene_fisura else 1 - probabilidad)
    }
    
    return resultado


def mostrar_resultado(resultado, ruta_imagen):
    """Muestra el resultado de forma visual"""
    
    print("\n" + "=" * 80)
    print("RESULTADO DE PREDICCIÓN")
    print("=" * 80)
    print(f"📷 Imagen: {Path(ruta_imagen).name}")
    print(f"📊 Clasificación: {resultado['clase']}")
    print(f"🎯 Confianza: {resultado['confianza']*100:.2f}%")
    print()
    print(f"Probabilidad de FISURA:    {resultado['probabilidad_fisura']*100:.2f}%")
    print(f"Probabilidad de SIN FISURA: {resultado['probabilidad_sin_fisura']*100:.2f}%")
    print("=" * 80)
    
    # Interpretación
    if resultado['tiene_fisura']:
        if resultado['confianza'] >= 0.9:
            print("⚠️  ALERTA: Se detectaron fisuras con ALTA confianza")
            print("   Recomendación: Inspección inmediata por ingeniero")
        elif resultado['confianza'] >= 0.7:
            print("⚠️  ADVERTENCIA: Posible presencia de fisuras")
            print("   Recomendación: Inspección manual recomendada")
        else:
            print("⚠️  PRECAUCIÓN: Indicios de fisuras (baja confianza)")
            print("   Recomendación: Verificar con más imágenes")
    else:
        if resultado['confianza'] >= 0.9:
            print("✅ SEGURO: No se detectaron fisuras")
            print("   Estado: Superficie en buen estado")
        elif resultado['confianza'] >= 0.7:
            print("✅ PROBABLEMENTE SEGURO: No se detectaron fisuras significativas")
            print("   Estado: Monitoreo preventivo recomendado")
        else:
            print("⚠️  INCIERTO: Clasificación poco confiable")
            print("   Recomendación: Tomar más imágenes o ajustar iluminación")
    
    print("=" * 80 + "\n")


def visualizar_prediccion(ruta_imagen, resultado):
    """Crea visualización con imagen y resultado"""
    
    # Cargar imagen
    img = Image.open(ruta_imagen)
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Imagen original
    ax1.imshow(img)
    ax1.set_title(f"Imagen: {Path(ruta_imagen).name}", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Resultado
    colores = ['#2ecc71', '#e74c3c']  # Verde, Rojo
    probabilidades = [
        resultado['probabilidad_sin_fisura'],
        resultado['probabilidad_fisura']
    ]
    etiquetas = ['Sin Fisura', 'Con Fisura']
    
    bars = ax2.barh(etiquetas, probabilidades, color=colores, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Resaltar clasificación
    clase_idx = 1 if resultado['tiene_fisura'] else 0
    bars[clase_idx].set_alpha(1.0)
    bars[clase_idx].set_edgecolor('gold')
    bars[clase_idx].set_linewidth(3)
    
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probabilidad', fontsize=12, fontweight='bold')
    ax2.set_title(
        f"Clasificación: {resultado['clase']}\nConfianza: {resultado['confianza']*100:.1f}%",
        fontsize=14,
        fontweight='bold',
        color=colores[clase_idx]
    )
    ax2.grid(axis='x', alpha=0.3)
    
    # Añadir valores en barras
    for i, (prob, label) in enumerate(zip(probabilidades, etiquetas)):
        ax2.text(prob + 0.02, i, f"{prob*100:.1f}%", 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Detectar fisuras en imágenes usando modelo entrenado"
    )
    parser.add_argument(
        '--imagen',
        type=str,
        required=True,
        help='Ruta a la imagen a analizar'
    )
    parser.add_argument(
        '--umbral',
        type=float,
        default=UMBRAL_DEFAULT,
        help=f'Umbral de clasificación (0-1, default: {UMBRAL_DEFAULT})'
    )
    parser.add_argument(
        '--visualizar',
        action='store_true',
        help='Mostrar visualización gráfica del resultado'
    )
    parser.add_argument(
        '--mostrar-imagen',
        action='store_true',
        help='Mostrar imagen original antes de procesar'
    )
    
    args = parser.parse_args()
    
    # Validar ruta de imagen
    ruta_imagen = Path(args.imagen)
    if not ruta_imagen.exists():
        print(f"❌ Error: No se encontró la imagen en: {ruta_imagen}")
        sys.exit(1)
    
    # Validar umbral
    if not 0 <= args.umbral <= 1:
        print(f"❌ Error: Umbral debe estar entre 0 y 1 (recibido: {args.umbral})")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("SISTEMA DE DETECCIÓN DE FISURAS ESTRUCTURALES")
    print("=" * 80)
    print(f"Modelo: MobileNetV2 + Transfer Learning")
    print(f"Precisión: 94.1% | Recall: 99.6% | AUC: 94.1%")
    print("=" * 80 + "\n")
    
    # Cargar modelo
    modelo = cargar_modelo_deteccion()
    
    # Preprocesar imagen
    print(f"📸 Procesando imagen: {ruta_imagen.name}...")
    img_array = preprocesar_imagen(ruta_imagen, mostrar=args.mostrar_imagen)
    
    # Predecir
    print(f"🔍 Analizando estructura...")
    resultado = predecir_fisura(modelo, img_array, umbral=args.umbral)
    
    # Mostrar resultado
    mostrar_resultado(resultado, ruta_imagen)
    
    # Visualización opcional
    if args.visualizar:
        visualizar_prediccion(ruta_imagen, resultado)


if __name__ == "__main__":
    main()
