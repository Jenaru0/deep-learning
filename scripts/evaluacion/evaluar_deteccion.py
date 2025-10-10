"""
SCRIPT DE EVALUACIÓN: Modelo de Detección de Fisuras
=====================================================

Este script evalúa un modelo entrenado de detección de fisuras
en el conjunto de test y genera reportes detallados.

Funcionalidades:
    - Carga el mejor modelo guardado
    - Evalúa en conjunto de test
    - Genera métricas completas (accuracy, precision, recall, F1, AUC)
    - Crea visualizaciones (matriz de confusión, ROC, PR curves)
    - Muestra ejemplos de predicciones correctas e incorrectas
    - Genera reporte JSON estructurado
    - Análisis de errores por categoría (Deck/Pavement/Wall)

Uso:
    python3 scripts/evaluacion/evaluar_deteccion.py [--modelo RUTA_MODELO]

Argumentos opcionales:
    --modelo: Ruta al modelo .h5 (default: mejor modelo en modelos/deteccion/)
    --visualizar: Número de ejemplos a visualizar (default: 20)
    --guardar: Directorio donde guardar resultados (default: resultados/visualizaciones/)

Autor: Jesus Naranjo (bajo supervisión de Claude 4.5)
Fecha: 9 de octubre de 2025
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow y Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Scikit-learn para métricas
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score
)

# Importar configuración
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    RUTA_DETECCION,
    RUTA_MODELO_DETECCION,
    RUTA_VIS,
    IMG_SIZE,
    BATCH_SIZE,
)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Establecer estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# FUNCIONES DE EVALUACIÓN
# =============================================================================

def cargar_modelo(ruta_modelo=None):
    """
    Carga el mejor modelo guardado.
    
    Args:
        ruta_modelo (str): Ruta explícita al modelo. Si es None, busca el mejor.
        
    Returns:
        keras.Model: Modelo cargado
    """
    if ruta_modelo is None:
        # Buscar el mejor modelo en el directorio
        modelos_dir = Path(RUTA_MODELO_DETECCION)
        
        # Buscar modelos .keras y .h5
        modelos_keras = list(modelos_dir.glob("*.keras"))
        modelos_h5 = list(modelos_dir.glob("*.h5"))
        modelos = modelos_keras + modelos_h5
        
        if not modelos:
            raise FileNotFoundError(f"No se encontraron modelos (.keras o .h5) en {modelos_dir}")
        
        # Preferir modelo final, luego best_stage2, luego cualquier "best"
        modelo_final = [m for m in modelos if "final" in m.name.lower()]
        modelo_stage2 = [m for m in modelos if "stage2" in m.name.lower() and "best" in m.name.lower()]
        modelos_best = [m for m in modelos if "best" in m.name.lower()]
        
        if modelo_final:
            ruta_modelo = modelo_final[0]
            print(f"📌 Usando modelo final: {ruta_modelo.name}")
        elif modelo_stage2:
            ruta_modelo = modelo_stage2[0]
            print(f"📌 Usando mejor modelo Stage 2: {ruta_modelo.name}")
        elif modelos_best:
            ruta_modelo = modelos_best[0]
            print(f"📌 Usando mejor modelo: {ruta_modelo.name}")
        else:
            # Tomar el más reciente
            ruta_modelo = max(modelos, key=lambda p: p.stat().st_mtime)
            print(f"📌 Usando modelo más reciente: {ruta_modelo.name}")
    
    print(f"\n📂 Cargando modelo desde: {ruta_modelo}")
    modelo = load_model(ruta_modelo)
    print(f"✅ Modelo cargado exitosamente")
    print(f"   Parámetros: {modelo.count_params():,}")
    
    return modelo


def crear_generador_test():
    """
    Crea generador de datos para el conjunto de test.
    
    Returns:
        DirectoryIterator: Generador de test
    """
    test_dir = Path(RUTA_DETECCION) / "test"
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # IMPORTANTE: No mezclar para análisis correcto
    )
    
    print(f"\n📊 Generador de test creado:")
    print(f"   Total de imágenes: {test_generator.samples}")
    print(f"   Clases: {test_generator.class_indices}")
    print(f"   Batches: {len(test_generator)}")
    
    return test_generator


def evaluar_metricas(modelo, test_gen):
    """
    Evalúa el modelo y calcula métricas completas.
    
    Args:
        modelo: Modelo de Keras
        test_gen: Generador de test
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    print(f"\n📈 Evaluando modelo en test set...")
    
    # Predicciones
    y_true = test_gen.classes
    y_pred_proba = modelo.predict(test_gen, verbose=1).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Nombres de clases
    class_names = list(test_gen.class_indices.keys())
    
    # Métricas básicas
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Especificidad (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Classification report detallado
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Construir diccionario de métricas
    metricas = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'classification_report': report,
        'total_samples': len(y_true),
        'class_distribution': {
            class_names[0]: int(np.sum(y_true == 0)),
            class_names[1]: int(np.sum(y_true == 1))
        }
    }
    
    return metricas, y_true, y_pred, y_pred_proba, cm, fpr, tpr, precision_curve, recall_curve


def imprimir_metricas(metricas):
    """
    Imprime métricas de forma legible en consola.
    
    Args:
        metricas (dict): Diccionario con métricas
    """
    print("\n" + "=" * 80)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 80)
    
    print(f"\n📊 Métricas Principales:")
    print(f"   Accuracy:    {metricas['accuracy']:.4f}")
    print(f"   Precision:   {metricas['precision']:.4f}")
    print(f"   Recall:      {metricas['recall']:.4f}")
    print(f"   F1-Score:    {metricas['f1_score']:.4f}")
    print(f"   Specificity: {metricas['specificity']:.4f}")
    print(f"   ROC-AUC:     {metricas['roc_auc']:.4f}")
    print(f"   Avg Precision: {metricas['average_precision']:.4f}")
    
    print(f"\n📊 Matriz de Confusión:")
    cm = metricas['confusion_matrix']
    print(f"   True Negatives:  {cm['tn']}")
    print(f"   False Positives: {cm['fp']}")
    print(f"   False Negatives: {cm['fn']}")
    print(f"   True Positives:  {cm['tp']}")
    
    print(f"\n📊 Distribución de Clases:")
    for clase, count in metricas['class_distribution'].items():
        porcentaje = count / metricas['total_samples'] * 100
        print(f"   {clase}: {count} ({porcentaje:.2f}%)")
    
    print("\n" + "=" * 80)


def visualizar_resultados(metricas, cm, fpr, tpr, precision_curve, recall_curve, output_dir):
    """
    Genera visualizaciones de los resultados.
    
    Args:
        metricas (dict): Diccionario con métricas
        cm (ndarray): Matriz de confusión
        fpr, tpr: Curva ROC
        precision_curve, recall_curve: Curva Precision-Recall
        output_dir (Path): Directorio de salida
    """
    print(f"\n📊 Generando visualizaciones...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Matriz de Confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Uncracked', 'Cracked'],
                yticklabels=['Uncracked', 'Cracked'])
    ax.set_title('Matriz de Confusión - Modelo de Detección', fontsize=14, fontweight='bold')
    ax.set_ylabel('Verdadero', fontsize=12)
    ax.set_xlabel('Predicho', fontsize=12)
    plt.tight_layout()
    cm_path = output_dir / 'confusion_matrix_eval.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Matriz de confusión: {cm_path}")
    
    # 2. Curva ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {metricas["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / 'roc_curve_eval.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Curva ROC: {roc_path}")
    
    # 3. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_curve, precision_curve, color='blue', lw=2,
            label=f'PR curve (AP = {metricas["average_precision"]:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = output_dir / 'precision_recall_curve_eval.png'
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Curva Precision-Recall: {pr_path}")
    
    # 4. Resumen de Métricas (barra)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
    metrics_values = [
        metricas['accuracy'],
        metricas['precision'],
        metricas['recall'],
        metricas['f1_score'],
        metricas['specificity'],
        metricas['roc_auc']
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax.barh(metrics_names, metrics_values, color=colors)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{metrics_values[i]:.3f}',
                ha='left', va='center', fontweight='bold')
    
    ax.set_xlim([0, 1.1])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Resumen de Métricas de Evaluación', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    summary_path = output_dir / 'metrics_summary_eval.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Resumen de métricas: {summary_path}")


def guardar_reporte_json(metricas, output_dir):
    """
    Guarda reporte completo en JSON.
    
    Args:
        metricas (dict): Diccionario con métricas
        output_dir (Path): Directorio de salida
    """
    output_dir = Path(output_dir)
    
    reporte = {
        'fecha_evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metricas': metricas
    }
    
    json_path = output_dir / 'evaluation_report_final.json'
    with open(json_path, 'w') as f:
        json.dump(reporte, f, indent=4)
    
    print(f"\n✅ Reporte JSON guardado: {json_path}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal de evaluación.
    """
    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Evaluar modelo de detección de fisuras')
    parser.add_argument('--modelo', type=str, default=None,
                        help='Ruta al modelo .h5 (default: mejor modelo en modelos/deteccion/)')
    parser.add_argument('--output', type=str, default=RUTA_VIS,
                        help='Directorio de salida (default: resultados/visualizaciones/)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EVALUACIÓN DE MODELO DE DETECCIÓN DE FISURAS")
    print("=" * 80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Cargar modelo
    modelo = cargar_modelo(args.modelo)
    
    # 2. Crear generador de test
    test_gen = crear_generador_test()
    
    # 3. Evaluar y calcular métricas
    metricas, y_true, y_pred, y_pred_proba, cm, fpr, tpr, prec_curve, rec_curve = evaluar_metricas(modelo, test_gen)
    
    # 4. Imprimir métricas
    imprimir_metricas(metricas)
    
    # 5. Generar visualizaciones
    visualizar_resultados(metricas, cm, fpr, tpr, prec_curve, rec_curve, args.output)
    
    # 6. Guardar reporte JSON
    guardar_reporte_json(metricas, args.output)
    
    print("\n" + "=" * 80)
    print("✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    print(f"Resultados guardados en: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
