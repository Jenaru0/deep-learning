#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Visualización de Predicciones del Modelo U-Net

Este script evalúa el modelo de segmentación en el conjunto de test,
generando visualizaciones comparativas entre:
- Imagen original
- Ground truth (máscara real)
- Predicción del modelo
- Overlay con mediciones

Autor: Sistema de Análisis de Fisuras
Fecha: 2025-10-10
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

# Añadir path raíz al PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import config
from scripts.analisis.medir_parametros import (
    ModeloSegmentacion,
    predecir_mascara,
    medir_ancho_fisura,
    detectar_orientacion,
    estimar_profundidad,
    configurar_gpu
)


def calcular_metricas_imagen(y_true, y_pred, threshold=0.5):
    """
    Calcula métricas de segmentación para una imagen individual.
    
    Args:
        y_true: Máscara ground truth (H, W) valores 0-255
        y_pred: Máscara predicha (H, W) valores 0-1
        threshold: Umbral para binarización
        
    Returns:
        dict: Diccionario con métricas (IoU, Dice, Accuracy, Precision, Recall)
    """
    # Binarizar máscaras
    y_true_bin = (y_true > 127).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    
    # Calcular intersección y unión
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    
    # IoU (Intersection over Union)
    iou = intersection / (union + 1e-7)
    
    # Dice Coefficient
    dice = (2.0 * intersection) / (np.sum(y_true_bin) + np.sum(y_pred_bin) + 1e-7)
    
    # Pixel Accuracy
    correct = np.sum(y_true_bin == y_pred_bin)
    total = y_true_bin.size
    accuracy = correct / total
    
    # Precision y Recall (para clase positiva: fisura)
    tp = intersection  # True Positives
    fp = np.sum(y_pred_bin) - intersection  # False Positives
    fn = np.sum(y_true_bin) - intersection  # False Negatives
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def crear_visualizacion_comparativa(
    imagen_path,
    gt_mask_path,
    pred_mask,
    metricas,
    parametros,
    output_path
):
    """
    Crea visualización comparativa de 6 paneles con análisis completo.
    
    Paneles:
    1. Imagen original
    2. Ground truth
    3. Predicción
    4. Overlay GT + Original
    5. Overlay Predicción + Original
    6. Diferencia (errores)
    """
    # Cargar imagen y ground truth
    imagen = np.array(Image.open(imagen_path).convert('RGB'))
    gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
    
    # Binarizar predicción
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Crear figura con 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f'Análisis Completo: {Path(imagen_path).name}\n' +
        f'IoU: {metricas["iou"]:.3f} | Dice: {metricas["dice"]:.3f} | ' +
        f'Accuracy: {metricas["accuracy"]:.3f}',
        fontsize=16, fontweight='bold'
    )
    
    # Panel 1: Imagen original
    axes[0, 0].imshow(imagen)
    axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel 2: Ground Truth
    axes[0, 1].imshow(imagen)
    axes[0, 1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Panel 3: Predicción
    axes[0, 2].imshow(imagen)
    axes[0, 2].imshow(pred_mask_bin, alpha=0.5, cmap='Blues')
    axes[0, 2].set_title('Predicción', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Panel 4: Solo Ground Truth (máscara)
    axes[1, 0].imshow(gt_mask, cmap='gray')
    axes[1, 0].set_title('GT Máscara', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Panel 5: Solo Predicción (máscara)
    axes[1, 1].imshow(pred_mask_bin, cmap='gray')
    axes[1, 1].set_title('Predicción Máscara', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Panel 6: Mapa de errores (TP, FP, FN)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    pred_bin = (pred_mask_bin > 127).astype(np.uint8)
    
    error_map = np.zeros((*gt_bin.shape, 3), dtype=np.uint8)
    # True Positives (verde)
    tp_mask = (gt_bin == 1) & (pred_bin == 1)
    error_map[tp_mask] = [0, 255, 0]
    # False Positives (azul)
    fp_mask = (gt_bin == 0) & (pred_bin == 1)
    error_map[fp_mask] = [0, 0, 255]
    # False Negatives (rojo)
    fn_mask = (gt_bin == 1) & (pred_bin == 0)
    error_map[fn_mask] = [255, 0, 0]
    
    axes[1, 2].imshow(error_map)
    axes[1, 2].set_title('Mapa de Errores', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Leyenda para mapa de errores
    tp_patch = mpatches.Patch(color='green', label='True Positive')
    fp_patch = mpatches.Patch(color='blue', label='False Positive')
    fn_patch = mpatches.Patch(color='red', label='False Negative')
    axes[1, 2].legend(
        handles=[tp_patch, fp_patch, fn_patch],
        loc='upper right',
        fontsize=10
    )
    
    # Añadir cuadro de texto con métricas y parámetros
    metrics_text = (
        f"MÉTRICAS DE SEGMENTACIÓN:\n"
        f"  • IoU: {metricas['iou']:.4f}\n"
        f"  • Dice: {metricas['dice']:.4f}\n"
        f"  • Accuracy: {metricas['accuracy']:.4f}\n"
        f"  • Precision: {metricas['precision']:.4f}\n"
        f"  • Recall: {metricas['recall']:.4f}\n"
        f"  • F1-Score: {metricas['f1']:.4f}\n\n"
        f"PARÁMETROS ESTRUCTURALES:\n"
        f"  • Ancho Prom: {parametros['ancho']['ancho_promedio_mm']:.2f} mm\n"
        f"  • Ancho Máx: {parametros['ancho']['ancho_maximo_mm']:.2f} mm\n"
        f"  • Área: {parametros['ancho']['area_total_mm2']:.0f} mm²\n"
        f"  • Orientación: {parametros['orientacion']['orientacion']}\n"
        f"  • Ángulo: {parametros['orientacion']['angulo_grados']:.1f}°\n"
        f"  • Profundidad: {parametros['profundidad']['profundidad_categoria']}"
    )
    
    plt.figtext(
        0.02, 0.02, metrics_text,
        fontsize=11,
        family='monospace',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def procesar_conjunto_test(
    modelo,
    test_dir,
    num_samples=None,
    modo='aleatorio',
    output_dir=None
):
    """
    Procesa el conjunto de test y genera visualizaciones.
    
    Args:
        modelo: Modelo U-Net cargado
        test_dir: Directorio con imágenes de test
        num_samples: Número de muestras a procesar (None = todas)
        modo: 'aleatorio', 'mejores', 'peores'
        output_dir: Directorio de salida
        
    Returns:
        dict: Estadísticas del procesamiento
    """
    # Obtener lista de imágenes
    imagenes_dir = test_dir / 'images'
    mascaras_dir = test_dir / 'masks'
    
    imagenes = sorted(list(imagenes_dir.glob('*.jpg')))
    
    print(f"\n📂 Encontradas {len(imagenes)} imágenes en test set")
    
    # Crear directorio de salida
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.RUTA_RESULTADOS) / 'visualizaciones' / f'predicciones_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar todas las imágenes para calcular métricas
    print("\n🔄 Calculando métricas en todas las imágenes...")
    resultados = []
    
    for img_path in tqdm(imagenes, desc="Procesando"):
        # Buscar máscara correspondiente
        mask_name = img_path.stem + '.png'
        mask_path = mascaras_dir / mask_name
        
        if not mask_path.exists():
            print(f"⚠️  Máscara no encontrada: {mask_name}")
            continue
        
        # Predecir
        pred_mask = predecir_mascara(modelo, str(img_path))
        
        # Cargar ground truth
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        
        # Calcular métricas
        metricas = calcular_metricas_imagen(gt_mask, pred_mask)
        
        # Medir parámetros
        parametros = {
            'ancho': medir_ancho_fisura(pred_mask),
            'orientacion': detectar_orientacion(pred_mask),
            'profundidad': estimar_profundidad(np.array(Image.open(img_path)), pred_mask)
        }
        
        resultados.append({
            'imagen': img_path.name,
            'imagen_path': str(img_path),
            'mask_path': str(mask_path),
            'pred_mask': pred_mask,
            'metricas': metricas,
            'parametros': parametros
        })
    
    # Ordenar según modo
    if modo == 'mejores':
        resultados.sort(key=lambda x: x['metricas']['iou'], reverse=True)
    elif modo == 'peores':
        resultados.sort(key=lambda x: x['metricas']['iou'])
    elif modo == 'aleatorio':
        random.shuffle(resultados)
    
    # Seleccionar muestras
    if num_samples is not None:
        muestras = resultados[:num_samples]
    else:
        muestras = resultados
    
    print(f"\n📊 Generando visualizaciones para {len(muestras)} imágenes...")
    
    # Generar visualizaciones
    for i, resultado in enumerate(tqdm(muestras, desc="Visualizando"), 1):
        output_path = output_dir / f"pred_{i:03d}_{resultado['imagen']}.png"
        
        crear_visualizacion_comparativa(
            resultado['imagen_path'],
            resultado['mask_path'],
            resultado['pred_mask'],
            resultado['metricas'],
            resultado['parametros'],
            output_path
        )
    
    # Calcular estadísticas globales
    iou_values = [r['metricas']['iou'] for r in resultados]
    dice_values = [r['metricas']['dice'] for r in resultados]
    acc_values = [r['metricas']['accuracy'] for r in resultados]
    
    estadisticas = {
        'total_imagenes': len(resultados),
        'iou': {
            'mean': float(np.mean(iou_values)),
            'std': float(np.std(iou_values)),
            'min': float(np.min(iou_values)),
            'max': float(np.max(iou_values)),
            'median': float(np.median(iou_values))
        },
        'dice': {
            'mean': float(np.mean(dice_values)),
            'std': float(np.std(dice_values)),
            'min': float(np.min(dice_values)),
            'max': float(np.max(dice_values)),
            'median': float(np.median(dice_values))
        },
        'accuracy': {
            'mean': float(np.mean(acc_values)),
            'std': float(np.std(acc_values)),
            'min': float(np.min(acc_values)),
            'max': float(np.max(acc_values)),
            'median': float(np.median(acc_values))
        }
    }
    
    # Guardar estadísticas
    stats_path = output_dir / 'estadisticas.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(estadisticas, f, indent=4, ensure_ascii=False)
    
    # Crear gráfico de distribución de métricas
    crear_graficos_distribucion(iou_values, dice_values, acc_values, output_dir)
    
    return estadisticas, output_dir


def crear_graficos_distribucion(iou_values, dice_values, acc_values, output_dir):
    """Crea histogramas de distribución de métricas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histograma IoU
    axes[0].hist(iou_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(iou_values), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(iou_values):.3f}')
    axes[0].axvline(np.median(iou_values), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(iou_values):.3f}')
    axes[0].set_xlabel('IoU', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].set_title('Distribución de IoU', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Histograma Dice
    axes[1].hist(dice_values, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(dice_values), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(dice_values):.3f}')
    axes[1].axvline(np.median(dice_values), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(dice_values):.3f}')
    axes[1].set_xlabel('Dice Coefficient', fontsize=12)
    axes[1].set_ylabel('Frecuencia', fontsize=12)
    axes[1].set_title('Distribución de Dice', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Histograma Accuracy
    axes[2].hist(acc_values, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[2].axvline(np.mean(acc_values), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(acc_values):.3f}')
    axes[2].axvline(np.median(acc_values), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(acc_values):.3f}')
    axes[2].set_xlabel('Accuracy', fontsize=12)
    axes[2].set_ylabel('Frecuencia', fontsize=12)
    axes[2].set_title('Distribución de Accuracy', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribucion_metricas.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualización de predicciones del modelo U-Net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Visualizar 10 imágenes aleatorias
  python visualizar_predicciones.py --num-imagenes 10
  
  # Visualizar las 20 mejores predicciones
  python visualizar_predicciones.py --num-imagenes 20 --modo mejores
  
  # Visualizar las 15 peores predicciones
  python visualizar_predicciones.py --num-imagenes 15 --modo peores
  
  # Procesar todas las imágenes del test set
  python visualizar_predicciones.py --all
        """
    )
    
    parser.add_argument(
        '--num-imagenes', '-n',
        type=int,
        default=10,
        help='Número de imágenes a visualizar (default: 10)'
    )
    
    parser.add_argument(
        '--modo', '-m',
        choices=['aleatorio', 'mejores', 'peores'],
        default='aleatorio',
        help='Modo de selección de imágenes (default: aleatorio)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Procesar todas las imágenes del test set'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Directorio de salida personalizado'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║       VISUALIZACIÓN DE PREDICCIONES - MODELO U-NET                  ║")
    print("║                                                                      ║")
    print("║  📊 Evaluación completa del conjunto de test                        ║")
    print("║  🎨 Generación de visualizaciones comparativas                      ║")
    print("║  📈 Estadísticas de métricas de segmentación                        ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")
    
    # Configurar GPU
    configurar_gpu()
    
    # Cargar modelo usando la clase ModeloSegmentacion
    print("🔄 Cargando modelo U-Net...")
    modelo = ModeloSegmentacion(config.RUTA_MODELO_SEGMENTACION)
    modelo.cargar()
    print(f"   ✅ Modelo cargado: {config.RUTA_MODELO_SEGMENTACION}")
    
    # Directorio de test (datos procesados, no dataset original)
    test_dir = Path(config.RUTA_SEGMENTACION) / 'test'
    
    # Configurar número de muestras
    num_samples = None if args.all else args.num_imagenes
    
    # Configurar output dir
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Procesar
    estadisticas, output_path = procesar_conjunto_test(
        modelo,
        test_dir,
        num_samples=num_samples,
        modo=args.modo,
        output_dir=output_dir
    )
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS GLOBALES DEL TEST SET")
    print("="*70)
    print(f"\n📁 Total de imágenes evaluadas: {estadisticas['total_imagenes']}")
    print(f"\n📏 IoU (Intersection over Union):")
    print(f"   • Media: {estadisticas['iou']['mean']:.4f} ± {estadisticas['iou']['std']:.4f}")
    print(f"   • Mediana: {estadisticas['iou']['median']:.4f}")
    print(f"   • Rango: [{estadisticas['iou']['min']:.4f}, {estadisticas['iou']['max']:.4f}]")
    
    print(f"\n🎯 Dice Coefficient:")
    print(f"   • Media: {estadisticas['dice']['mean']:.4f} ± {estadisticas['dice']['std']:.4f}")
    print(f"   • Mediana: {estadisticas['dice']['median']:.4f}")
    print(f"   • Rango: [{estadisticas['dice']['min']:.4f}, {estadisticas['dice']['max']:.4f}]")
    
    print(f"\n✅ Accuracy:")
    print(f"   • Media: {estadisticas['accuracy']['mean']:.4f} ± {estadisticas['accuracy']['std']:.4f}")
    print(f"   • Mediana: {estadisticas['accuracy']['median']:.4f}")
    print(f"   • Rango: [{estadisticas['accuracy']['min']:.4f}, {estadisticas['accuracy']['max']:.4f}]")
    
    print(f"\n💾 Resultados guardados en:")
    print(f"   {output_path}")
    print("\n" + "="*70)
    print("✅ VISUALIZACIÓN COMPLETADA")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
