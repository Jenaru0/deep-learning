"""
TAREA #4: Entrenamiento de modelo de detección de fisuras
==========================================================

Este script entrena un modelo MobileNetV2 para clasificación binaria
(cracked vs uncracked) sobre el dataset SDNET2018.

Arquitectura:
    - MobileNetV2 preentrenado en ImageNet (transfer learning)
    - Fine-tuning en dos etapas (freeze base → full training)
    - Clasificación binaria con capa Dense(1, sigmoid)

Dataset:
    - SDNET2018: 56,092 imágenes
    - Train: 39,261 | Val: 8,414 | Test: 8,417
    - Clases desbalanceadas: ~15% cracked, ~85% uncracked

Estrategia:
    - Data augmentation para aumentar variabilidad
    - Class weights para manejar desbalance de clases
    - Callbacks para early stopping y guardado del mejor modelo
    - Evaluación completa con métricas, gráficos y reportes JSON

Por qué MobileNetV2:
    - Más ligero que EfficientNetB0 (3.5M vs 5.3M parámetros)
    - Mejor para detectar objetos pequeños (fisuras finas)
    - Compatible con Keras 3.x sin bugs de compatibilidad
    - Entrenamiento ~20% más rápido
    - Excelente para deployment en edge devices

Optimizaciones para Keras 3.x + TensorFlow 2.17:
    - tf.data.Dataset con prefetch(AUTOTUNE) para pipeline asíncrono
    - Mixed precision (float16) para aprovechar Tensor Cores RTX
    - XLA JIT compilation para optimización de grafos
    - Límite VRAM 3.5GB para GPUs de 4GB (RTX 2050)

Compatibilidad:
    - TensorFlow >= 2.15 con Keras 3.x
    - CUDA 12.x + cuDNN 8.9+
    - Python 3.10+

Autor: Jesus Naranjo
Fecha: Octubre 2025
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# TensorFlow y Keras (usar tensorflow.keras directamente)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model, Input

# Scikit-learn para métricas
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Importar configuración del proyecto
# Asumiendo que ejecutas desde la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    RUTA_DETECCION,
    RUTA_MODELO_DETECCION,
    RUTA_VIS,
    RUTA_REPORTES,
    IMG_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    AUGMENTATION_CONFIG
)

# Crear alias para compatibilidad
DATOS_PROCESADOS_DETECCION = Path(RUTA_DETECCION)
MODELOS_DETECCION = Path(RUTA_MODELO_DETECCION)
RESULTADOS_VISUALIZACIONES = Path(RUTA_VIS)
REPORTES_DIR = Path(RUTA_REPORTES)
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE

# Parámetros de entrenamiento por etapas
EPOCHS_STAGE1 = 10  # Primera etapa: base congelada
EPOCHS_STAGE2 = 40  # Segunda etapa: fine-tuning completo
LEARNING_RATE_STAGE1 = 1e-3  # Learning rate alto para stage 1
LEARNING_RATE_STAGE2 = 1e-4  # Learning rate bajo para fine-tuning

# OPTIMIZACIÓN: Aumentar workers para I/O paralelo (aprovecha CPU 12 núcleos)
# WSL tiene acceso limitado, pero podemos usar 4-6 workers sin problemas
NUM_IO_THREADS = 6  # Para I/O de disco paralelo

# =============================================================================
# CONFIGURACIÓN Y SEMILLAS
# =============================================================================

# Establecer semillas para reproducibilidad
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configurar GPU de forma óptima
print("=" * 80)
print("CONFIGURACIÓN DEL ENTORNO")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")

# CRÍTICO: Listar TODAS las GPUs detectadas (Intel + NVIDIA)
all_gpus = tf.config.list_physical_devices('GPU')
print(f"\n🔍 GPUs detectadas por TensorFlow:")
for idx, gpu in enumerate(all_gpus):
    print(f"   GPU {idx}: {gpu.name} ({gpu.device_type})")

print(f"\nBuilt with CUDA: {tf.test.is_built_with_cuda()}")
print("=" * 80)

# =============================================================================
# FORZAR USO EXCLUSIVO DE NVIDIA RTX 2050 (ignorar Intel iGPU)
# =============================================================================
# =============================================================================
# FORZAR USO EXCLUSIVO DE NVIDIA RTX 2050 (ignorar Intel iGPU)
# =============================================================================
gpus = all_gpus
if gpus:
    try:
        # Si hay múltiples GPUs, seleccionar SOLO la NVIDIA (típicamente la última)
        # Orden común: [Intel iGPU, NVIDIA dGPU]
        nvidia_gpu = None
        
        # Intentar identificar la NVIDIA por nombre
        for gpu in gpus:
            gpu_name = gpu.name.lower()
            if 'nvidia' in gpu_name or 'geforce' in gpu_name or 'rtx' in gpu_name:
                nvidia_gpu = gpu
                break
        
        # Si no se encuentra por nombre, usar la última GPU (generalmente la dedicada)
        if nvidia_gpu is None and len(gpus) > 1:
            nvidia_gpu = gpus[-1]  # Última GPU = generalmente NVIDIA
            print(f"⚠️ No se detectó 'nvidia' en nombres - usando última GPU: {nvidia_gpu.name}")
        elif nvidia_gpu is None:
            nvidia_gpu = gpus[0]  # Solo hay una GPU
        
        # FORZAR TensorFlow a usar SOLO la NVIDIA RTX 2050
        tf.config.set_visible_devices([nvidia_gpu], 'GPU')
        print(f"\n✅ FORZANDO uso EXCLUSIVO de: {nvidia_gpu.name}")
        print(f"   (Intel iGPU IGNORADA - solo usamos NVIDIA RTX 2050)")
        
        # Configurar límite de memoria a 3.5GB (dejar 500MB para sistema)
        tf.config.set_logical_device_configuration(
            nvidia_gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
        )
        print(f"✅ GPU configurada con límite de 3500 MB VRAM")
        
    except RuntimeError as e:
        print(f"⚠️ Error configurando GPU: {e}")
        # Fallback: usar memory growth en la primera GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✅ Memory growth habilitado para GPU")
        except RuntimeError as e2:
            print(f"⚠️ Error con memory growth: {e2}")
else:
    print("❌ No se detectó GPU - usando CPU (será MUY lento)")
    print("   Verifica drivers NVIDIA y CUDA installation")

# Optimización 2: Habilitar mixed precision (usa Tensor Cores de RTX 2050)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision (float16) activado - 2x más rápido en GPU")

# Optimización 3: Habilitar XLA (Accelerated Linear Algebra)
tf.config.optimizer.set_jit(True)
print("✅ XLA JIT compilation habilitado")

# Optimización 4: Configurar threading para I/O paralelo
tf.config.threading.set_inter_op_parallelism_threads(NUM_IO_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_IO_THREADS)
print(f"✅ Threading configurado: {NUM_IO_THREADS} threads para I/O paralelo")

print("=" * 80)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def calcular_class_weights(train_dir):
    """
    Calcula pesos de clases para manejar el desbalance.
    
    La fórmula es: weight[class] = total_samples / (n_classes * samples_per_class)
    Esto asigna mayor peso a la clase minoritaria (cracked).
    
    Args:
        train_dir (Path): Directorio de entrenamiento con subdirectorios por clase
        
    Returns:
        dict: Diccionario {class_index: weight}
    """
    print("\n📊 Calculando pesos de clases...")
    
    # Contar imágenes por clase
    cracked_count = len(list((train_dir / "cracked").glob("*.jpg")))
    uncracked_count = len(list((train_dir / "uncracked").glob("*.jpg")))
    total = cracked_count + uncracked_count
    
    print(f"   Cracked: {cracked_count} imágenes ({cracked_count/total*100:.2f}%)")
    print(f"   Uncracked: {uncracked_count} imágenes ({uncracked_count/total*100:.2f}%)")
    
    # Calcular pesos (asumiendo cracked=1, uncracked=0 en orden alfabético)
    # ImageDataGenerator asigna índices alfabéticamente: cracked=0, uncracked=1
    weight_cracked = total / (2 * cracked_count)
    weight_uncracked = total / (2 * uncracked_count)
    
    class_weights = {
        0: weight_cracked,    # cracked (minoritaria)
        1: weight_uncracked   # uncracked (mayoritaria)
    }
    
    print(f"   Class weights: cracked={weight_cracked:.4f}, uncracked={weight_uncracked:.4f}")
    print(f"   ✅ Clase minoritaria (cracked) tiene {weight_cracked/weight_uncracked:.2f}x más peso")
    
    return class_weights


def crear_generadores_datos(train_dir, val_dir, test_dir, augmentation=True):
    """
    Crea generadores de datos con/sin augmentation.
    
    Data augmentation solo se aplica al conjunto de entrenamiento.
    Val y test usan solo normalización (rescale) para evaluación justa.
    
    Args:
        train_dir (Path): Directorio de entrenamiento
        val_dir (Path): Directorio de validación
        test_dir (Path): Directorio de test
        augmentation (bool): Si aplicar data augmentation al train set
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    print("\n🔄 Creando generadores de datos...")
    
    # Generador de entrenamiento CON augmentation
    if augmentation:
        print("   ✅ Data augmentation activado para training set")
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalización [0, 1]
            rotation_range=AUGMENTATION_CONFIG["rotation_range"],
            width_shift_range=AUGMENTATION_CONFIG["width_shift_range"],
            height_shift_range=AUGMENTATION_CONFIG["height_shift_range"],
            horizontal_flip=AUGMENTATION_CONFIG["horizontal_flip"],
            zoom_range=AUGMENTATION_CONFIG["zoom_range"],
            brightness_range=AUGMENTATION_CONFIG["brightness_range"],
            fill_mode='nearest'  # Rellenar píxeles vacíos con el más cercano
        )
    else:
        # Solo normalización (para debugging o comparación)
        print("   ⚠️ Data augmentation DESACTIVADO")
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generadores de validación y test SOLO con normalización
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Crear generadores desde directorios
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Clasificación binaria
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # No mezclar validación para reproducibilidad
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # No mezclar test para evaluación correcta
    )
    
    print(f"   Train batches: {len(train_generator)}")
    print(f"   Val batches: {len(val_generator)}")
    print(f"   Test batches: {len(test_generator)}")
    print(f"   Mapeo de clases: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator


def construir_modelo(input_shape=(224, 224, 3), trainable_base=False):
    """
    Construye el modelo MobileNetV2 para clasificación binaria.
    
    Arquitectura:
        - Base: MobileNetV2 preentrenado en ImageNet
        - GlobalAveragePooling2D: Reduce feature maps a vector 1D
        - Dropout(0.3): Regularización para evitar overfitting
        - Dense(256, ReLU): Capa fully-connected para aprendizaje de alto nivel
        - Dropout(0.3): Más regularización
        - Dense(1, Sigmoid): Salida binaria [0, 1]
    
    MobileNetV2 vs EfficientNetB0:
        - Más ligero: 3.5M vs 5.3M parámetros
        - Más rápido: ~20% menos tiempo por época
        - Mejor para objetos pequeños (fisuras finas)
        - Compatible con Keras 3.x sin bugs
        - Excelente para edge deployment
    
    Args:
        input_shape (tuple): Tamaño de entrada (height, width, channels)
        trainable_base (bool): Si entrenar la base MobileNetV2
        
    Returns:
        Model: Modelo compilado
    """
    print("\n🏗️ Construyendo modelo MobileNetV2...")
    
    # Cargar MobileNetV2 preentrenado en ImageNet (sin top classification layer)
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        alpha=1.0  # Width multiplier (1.0 = modelo completo)
    )
    
    # Congelar/descongelar base según estrategia de entrenamiento
    base_model.trainable = trainable_base
    
    if trainable_base:
        print(f"   ✅ Base MobileNetV2 ENTRENABLE (fine-tuning completo)")
    else:
        print(f"   🔒 Base MobileNetV2 CONGELADA (solo entrenar top layers)")
    
    # Construir modelo completo
    inputs = Input(shape=input_shape)
    
    # Base convolucional (feature extraction)
    x = base_model(inputs, training=False)
    
    # Global Average Pooling: convierte feature maps a vector 1D
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout para regularización
    x = layers.Dropout(0.3)(x)
    
    # Capa fully-connected
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    
    # Más dropout
    x = layers.Dropout(0.3)(x)
    
    # Capa de salida: 1 neurona con sigmoid para probabilidad binaria
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Crear modelo
    model = Model(inputs, outputs, name='MobileNetV2_CrackDetection')
    
    # Resumen de arquitectura
    print(f"\n   Parámetros totales: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    print(f"   Parámetros entrenables: {trainable_params:,}")
    print(f"   Parámetros no entrenables: {non_trainable_params:,}")
    
    return model


def compilar_modelo(model, learning_rate):
    """
    Compila el modelo con optimizer, loss y métricas.
    
    - Optimizer: Adam (adaptive learning rate)
    - Loss: Binary Crossentropy (para clasificación binaria)
    - Métricas: Accuracy, Precision, Recall, AUC
    
    Args:
        model (keras.Model): Modelo a compilar
        learning_rate (float): Tasa de aprendizaje
    """
    print(f"\n⚙️ Compilando modelo con learning_rate={learning_rate}...")
    
    # Para mixed precision, necesitamos usar loss scaling
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Si estamos usando mixed precision, envolver optimizer con LossScaleOptimizer
    if mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        print("   ✅ LossScaleOptimizer activado para mixed precision")
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("   ✅ Modelo compilado exitosamente")


def entrenar_modelo(model, train_gen, val_gen, class_weights, epochs, stage_name, checkpoint_path):
    """
    Entrena el modelo con callbacks para monitoreo y guardado.
    
    ACTUALIZADO para Keras 3.x:
        - Convierte ImageDataGenerator a tf.data.Dataset para paralelismo
        - Usa prefetch(AUTOTUNE) para pipeline asíncrono CPU-GPU
        - Compatible con mixed precision y XLA JIT
    
    Callbacks:
        - ModelCheckpoint: Guarda el mejor modelo según val_loss
        - EarlyStopping: Detiene si no mejora en 10 épocas
        - ReduceLROnPlateau: Reduce learning rate si se estanca
    
    Args:
        model (Model): Modelo a entrenar
        train_gen: Generador de entrenamiento
        val_gen: Generador de validación
        class_weights (dict): Pesos de clases
        epochs (int): Número de épocas
        stage_name (str): Nombre de la etapa (para logging)
        checkpoint_path (Path): Ruta donde guardar el mejor modelo
        
    Returns:
        History: Historial de entrenamiento
    """
    print(f"\n🚀 Iniciando entrenamiento - {stage_name}")
    print(f"   Épocas: {epochs}")
    print(f"   Checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Callbacks
    callbacks = [
        # Guardar el mejor modelo según val_loss
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        
        # Early stopping: detener si no mejora en 10 épocas
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,  # Restaurar pesos del mejor modelo
            verbose=1
        ),
        
        # Reducir learning rate si se estanca (no mejora en 5 épocas)
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reducir a la mitad
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # =========================================================================
    # MIGRACIÓN A tf.data.Dataset PARA KERAS 3.x + OPTIMIZACIONES GPU
    # =========================================================================
    # Convertir generadores a tf.data.Dataset con optimizaciones agresivas
    print("   🔄 Convirtiendo generadores a tf.data.Dataset con optimizaciones...")
    
    # Determinar output signature según el generador
    # ImageDataGenerator con class_mode='binary' produce (batch_images, batch_labels)
    output_signature = (
        tf.TensorSpec(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    
    # OPTIMIZACIÓN 1: Usar num_parallel_calls para paralelización CPU
    NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
    PREFETCH_BUFFER = 3  # Prefetch 3 batches (agresivo para GPU rápida)
    
    # Crear datasets con pipeline AGRESIVAMENTE optimizado
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=output_signature
    ).prefetch(PREFETCH_BUFFER)  # Prefetch múltiples batches
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=output_signature
    ).prefetch(PREFETCH_BUFFER)
    
    # OPTIMIZACIÓN 2: Cachear validación en RAM (acelera validación entre épocas)
    # Comentar si no tienes suficiente RAM (necesitas ~3GB libres)
    # val_dataset = val_dataset.cache()
    
    print("   ✅ Datasets creados con prefetch agresivo - pipeline optimizado")
    print(f"   ⚡ Prefetch buffer: {PREFETCH_BUFFER} batches")
    
    # =========================================================================
    # ENTRENAMIENTO CON API KERAS 3.x (sin workers/use_multiprocessing)
    # =========================================================================
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,  # Aplicar pesos de clases
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ {stage_name} completado")
    
    return history


def evaluar_modelo(model, test_gen, output_dir):
    """
    Evalúa el modelo en el conjunto de test y genera reportes completos.
    
    Genera:
        - Métricas de clasificación (accuracy, precision, recall, f1-score)
        - Matriz de confusión
        - Curva ROC y AUC
        - Curva Precision-Recall
        - Reporte JSON con todas las métricas
    
    Args:
        model (keras.Model): Modelo entrenado
        test_gen: Generador de test
        output_dir (Path): Directorio donde guardar resultados
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    print("\n📊 Evaluando modelo en conjunto de test...")
    print("=" * 80)
    
    # Obtener predicciones
    print("   Generando predicciones...")
    y_true = test_gen.classes
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Nombres de clases (cracked=0, uncracked=1)
    class_names = list(test_gen.class_indices.keys())
    
    # Métricas básicas
    print("\n   Calculando métricas...")
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_gen, verbose=0)
    
    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Imprimir resultados
    print("\n" + "=" * 80)
    print("RESULTADOS EN TEST SET")
    print("=" * 80)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"\nMatriz de confusión:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("=" * 80)
    
    # Guardar métricas en JSON
    metrics_dict = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'total_samples': len(y_true),
        'class_distribution': {
            class_names[0]: int(np.sum(y_true == 0)),
            class_names[1]: int(np.sum(y_true == 1))
        }
    }
    
    json_path = output_dir / 'evaluation_report.json'
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\n✅ Métricas guardadas en: {json_path}")
    
    # Visualizaciones
    print("\n📈 Generando visualizaciones...")
    
    # 1. Matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - Test Set')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Matriz de confusión guardada: {cm_path}")
    
    # 2. Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / 'roc_curve.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Curva ROC guardada: {roc_path}")
    
    # 3. Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = output_dir / 'precision_recall_curve.png'
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Curva Precision-Recall guardada: {pr_path}")
    
    return metrics_dict


def plot_training_history(history_stage1, history_stage2, output_dir):
    """
    Grafica el historial de entrenamiento de ambas etapas.
    
    Genera gráficos de:
        - Loss (train vs val) para ambas etapas
        - Accuracy (train vs val) para ambas etapas
        - Otras métricas (precision, recall, AUC)
    
    Args:
        history_stage1 (keras.callbacks.History): Historial de Stage 1
        history_stage2 (keras.callbacks.History): Historial de Stage 2
        output_dir (Path): Directorio donde guardar gráficos
    """
    print("\n📊 Generando gráficos de entrenamiento...")
    
    # Combinar historiales
    def get_metric(history, metric_name):
        return history.history.get(metric_name, [])
    
    # Concatenar métricas de ambas etapas
    epochs_stage1 = len(get_metric(history_stage1, 'loss'))
    epochs_stage2 = len(get_metric(history_stage2, 'loss'))
    
    metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Concatenar train y val de ambas etapas
        train_metric = get_metric(history_stage1, metric) + get_metric(history_stage2, metric)
        val_metric = get_metric(history_stage1, f'val_{metric}') + get_metric(history_stage2, f'val_{metric}')
        
        epochs = range(1, len(train_metric) + 1)
        
        # Plot
        ax.plot(epochs, train_metric, 'b-', label=f'Train {metric}', linewidth=2)
        ax.plot(epochs, val_metric, 'r-', label=f'Val {metric}', linewidth=2)
        
        # Línea vertical separando etapas
        if epochs_stage1 > 0:
            ax.axvline(x=epochs_stage1, color='green', linestyle='--', 
                       label='Stage 1 → Stage 2', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} durante entrenamiento')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Ocultar subplot sobrante
    axes[-1].axis('off')
    
    plt.tight_layout()
    history_path = output_dir / 'training_history.png'
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Historial de entrenamiento guardado: {history_path}")
    
    # Guardar historial en JSON
    history_dict = {
        'stage1': {
            'epochs': epochs_stage1,
            'history': {k: [float(v) for v in vals] for k, vals in history_stage1.history.items()}
        },
        'stage2': {
            'epochs': epochs_stage2,
            'history': {k: [float(v) for v in vals] for k, vals in history_stage2.history.items()}
        }
    }
    
    json_path = output_dir / 'training_history.json'
    with open(json_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"   ✅ Historial JSON guardado: {json_path}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta el pipeline completo de entrenamiento.
    
    Pipeline:
        1. Preparar directorios
        2. Crear generadores de datos
        3. Calcular class weights
        4. STAGE 1: Entrenar solo top layers (base congelada)
        5. STAGE 2: Fine-tuning completo (descongelar base)
        6. Evaluar en test set
        7. Generar reportes y visualizaciones
    """
    print("\n" + "=" * 80)
    print("TAREA #4: ENTRENAMIENTO DE MODELO DE DETECCIÓN")
    print("=" * 80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # 1. PREPARAR DIRECTORIOS
    # -------------------------------------------------------------------------
    
    train_dir = DATOS_PROCESADOS_DETECCION / "train"
    val_dir = DATOS_PROCESADOS_DETECCION / "val"
    test_dir = DATOS_PROCESADOS_DETECCION / "test"
    
    # Verificar que existan
    for directory in [train_dir, val_dir, test_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    
    # Crear directorios de salida
    MODELOS_DETECCION.mkdir(parents=True, exist_ok=True)
    RESULTADOS_VISUALIZACIONES.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Directorios:")
    print(f"   Train: {train_dir}")
    print(f"   Val: {val_dir}")
    print(f"   Test: {test_dir}")
    print(f"   Modelos: {MODELOS_DETECCION}")
    print(f"   Visualizaciones: {RESULTADOS_VISUALIZACIONES}")
    
    # -------------------------------------------------------------------------
    # 2. CREAR GENERADORES DE DATOS
    # -------------------------------------------------------------------------
    
    train_gen, val_gen, test_gen = crear_generadores_datos(
        train_dir, val_dir, test_dir, augmentation=True
    )
    
    # -------------------------------------------------------------------------
    # 3. CALCULAR CLASS WEIGHTS
    # -------------------------------------------------------------------------
    
    class_weights = calcular_class_weights(train_dir)
    
    # -------------------------------------------------------------------------
    # 4. STAGE 1: ENTRENAR TOP LAYERS (BASE CONGELADA)
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STAGE 1: ENTRENAMIENTO CON BASE CONGELADA")
    print("=" * 80)
    print("Estrategia: Entrenar solo capas superiores (dense layers)")
    print("Objetivo: Adaptar clasificador a dataset de fisuras")
    print("=" * 80)
    
    # Construir modelo con base congelada
    model = construir_modelo(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        trainable_base=False  # Congelar MobileNetV2
    )
    
    # Compilar con learning rate alto (stage 1)
    compilar_modelo(model, learning_rate=LEARNING_RATE_STAGE1)
    
    # Entrenar
    checkpoint_stage1 = MODELOS_DETECCION / "mobilenetv2_stage1.h5"
    history_stage1 = entrenar_modelo(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights,
        epochs=EPOCHS_STAGE1,
        stage_name="STAGE 1",
        checkpoint_path=checkpoint_stage1
    )
    
    # -------------------------------------------------------------------------
    # 5. STAGE 2: FINE-TUNING COMPLETO (DESCONGELAR BASE)
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STAGE 2: FINE-TUNING COMPLETO")
    print("=" * 80)
    print("Estrategia: Descongelar base MobileNetV2 y entrenar todo")
    print("Objetivo: Afinar pesos preentrenados para fisuras específicamente")
    print("=" * 80)
    
    # Descongelar base
    for layer in model.layers:
        if hasattr(layer, 'trainable'):
            layer.trainable = True
    
    print(f"   ✅ Base MobileNetV2 descongelada")
    
    # Recompilar con learning rate bajo (stage 2) para fine-tuning
    compilar_modelo(model, learning_rate=LEARNING_RATE_STAGE2)
    
    # Entrenar
    checkpoint_stage2 = MODELOS_DETECCION / "mobilenetv2_best.h5"
    history_stage2 = entrenar_modelo(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights,
        epochs=EPOCHS_STAGE2,
        stage_name="STAGE 2",
        checkpoint_path=checkpoint_stage2
    )
    
    # -------------------------------------------------------------------------
    # 6. CARGAR MEJOR MODELO Y EVALUAR
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("CARGANDO MEJOR MODELO PARA EVALUACIÓN")
    print("=" * 80)
    
    # Cargar el mejor modelo guardado
    best_model = tf.keras.models.load_model(checkpoint_stage2)
    print(f"   ✅ Modelo cargado desde: {checkpoint_stage2}")
    
    # Evaluar en test set
    metrics = evaluar_modelo(
        model=best_model,
        test_gen=test_gen,
        output_dir=RESULTADOS_VISUALIZACIONES
    )
    
    # -------------------------------------------------------------------------
    # 7. GENERAR REPORTES Y VISUALIZACIONES
    # -------------------------------------------------------------------------
    
    # Gráficos de entrenamiento
    plot_training_history(history_stage1, history_stage2, RESULTADOS_VISUALIZACIONES)
    
    # -------------------------------------------------------------------------
    # 8. RESUMEN FINAL
    # -------------------------------------------------------------------------
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"Tiempo total: {elapsed_time/60:.2f} minutos")
    print(f"Mejor modelo guardado en: {checkpoint_stage2}")
    print(f"Visualizaciones en: {RESULTADOS_VISUALIZACIONES}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test AUC: {metrics['test_auc']:.4f}")
    print("=" * 80)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Guardar resumen final
    summary = {
        'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tiempo_total_minutos': elapsed_time / 60,
        'configuracion': {
            'img_height': IMG_HEIGHT,
            'img_width': IMG_WIDTH,
            'batch_size': BATCH_SIZE,
            'epochs_stage1': EPOCHS_STAGE1,
            'epochs_stage2': EPOCHS_STAGE2,
            'learning_rate_stage1': LEARNING_RATE_STAGE1,
            'learning_rate_stage2': LEARNING_RATE_STAGE2,
            'random_seed': RANDOM_SEED,
            'augmentation': AUGMENTATION_CONFIG
        },
        'class_weights': class_weights,
        'resultados_test': metrics,
        'archivos_generados': {
            'modelo_stage1': str(checkpoint_stage1),
            'modelo_final': str(checkpoint_stage2),
            'training_history': str(RESULTADOS_VISUALIZACIONES / 'training_history.json'),
            'evaluation_report': str(RESULTADOS_VISUALIZACIONES / 'evaluation_report.json')
        }
    }
    
    summary_path = REPORTES_DIR / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✅ Resumen completo guardado en: {summary_path}")
    print("\n🎉 ¡TAREA #4 COMPLETADA EXITOSAMENTE!")


if __name__ == "__main__":
    main()
