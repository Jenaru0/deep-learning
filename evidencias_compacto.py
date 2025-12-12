"""
Versión compacta: todas las tablas en una sola pantalla
Para captura única
"""

import json
from pathlib import Path
from datetime import datetime

print("\n" + "="*100)
print("EVIDENCIAS DE RESULTADOS - Sistema de Detección de Fisuras".center(100))
print("Universidad Nacional de Cañete - Deep Learning 2025".center(100))
print("="*100 + "\n")

# TABLA 1
print("TABLA 1: Modelo de Detección (SDNET2018)")
print("-" * 60)
data = json.load(open("resultados/visualizaciones/evaluation_report_final.json"))
m = data['metricas']
print(f"Accuracy: {m['accuracy']*100:.2f}%  |  Precision: {m['precision']*100:.2f}%  |  Recall: {m['recall']*100:.2f}%")
print(f"F1-Score: {m['f1_score']*100:.2f}%  |  AUC-ROC: {m['roc_auc']*100:.2f}%")
print(f"Muestras: {m['total_samples']}  |  Fecha: {data['fecha_evaluacion']}\n")

# TABLA 2
print("TABLA 2: Hiperparámetros U-Net")
print("-" * 60)
print("Arquitectura: U-Net lite (4 niveles, 32-256 filtros)  |  Parámetros: ~1.95M")
print("Entrada: 128×128×3  |  Batch: 4  |  LR: 1e-4  |  Pérdida: BCE+Dice")
print("Épocas: 50 planificadas / 36 ejecutadas (early stopping)  |  GPU: RTX 2050\n")

# TABLA 3
print("TABLA 3: Entrenamiento Época 36")
print("-" * 60)
data = json.load(open("modelos/segmentacion/resultados_test_20251010_051114.json"))
print(f"IoU: {data['test_iou']*100:.2f}%  |  Dice: {data['test_dice']*100:.2f}%  |  Accuracy: {data['test_accuracy']*100:.2f}%")
print(f"Tiempo: {data['tiempo_entrenamiento_min']:.2f} min  |  Timestamp: {data['timestamp']}\n")

# TABLA 4
print("TABLA 4: Test CRACK500 (n=1,124)")
print("-" * 60)
print("IoU:  Media=60.51% | DE=19.63% | Min=0% | Max=90.25% | Mediana=65.43%")
print("Dice: Media=73.04% | DE=19.62% | Min=0% | Max=94.87% | Mediana=79.10%")
print("Acc:  Media=97.38% | DE=1.75%  | Min=86.25% | Max=100% | Mediana=97.73%\n")

# TABLA 5
print("TABLA 5: Rendimiento Temporal")
print("-" * 60)
print("Evaluación (1,124 imgs): ~2 min 38 s  →  ~7.6 img/s")
print("Visualizaciones (10 imgs): ~19 s  →  ~1.9 s/img\n")

# TABLA 6
print("TABLA 6: Parámetros Estructurales")
print("-" * 60)
print("Ancho promedio/máximo: Distance transform  |  Longitud: Esqueleto")
print("Orientación: PCA/Hough  |  Área: Conteo píxeles calibrados\n")

# TABLA 7
print("TABLA 7: Comparativa Estado del Arte")
print("-" * 60)
print("Este trabajo (U-Net lite 1.95M):  IoU=60.5%  Dice=73.0%  [CRACK500]")
print("Yang et al. (U-Net std 31M):      IoU=~65%   Dice=~75%   [CRACK500]")
print("Li et al. (Mini-UNet):             IoU=58%    Dice=70%    [Propio]")
print("Manjunatha (CrackDenseLinkNet):   IoU=68%    Dice=78%    [Propio+luz]\n")

print("="*100)
print(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*100 + "\n")
