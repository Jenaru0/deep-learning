"""
ENTRENAMIENTO: Modelo U-Net para Segmentación de Fisuras
=========================================================

Este script entrena un modelo U-Net para segmentación semántica de fisuras
sobre el dataset CRACK500.

Arquitectura:
    - U-Net Lite (optimizada para RTX 2050 4GB)
    - Input: 128x128x3 (RGB) - reducido de 256x256 para ahorrar VRAM
    - Output: 128x128x1 (máscara binaria)
    - Filtros: 32→64→128→256 (en vez de 64→128→256→512→1024)
    - Parámetros: ~2M (en vez de 31M) = 93% reducción
    
Dataset:
    - CRACK500: 3,368 pares imagen-máscara
    - Train: 1,896 | Val: 348 | Test: 1,124
    - Máscaras binarias (0=fondo, 1=fisura)

Optimizaciones para RTX 2050 (4GB VRAM):
    - Mixed Precision FP16 (reduce VRAM 40%)
    - XLA JIT Compilation (+20% velocidad)
    - Batch size = 4 (muy conservador)
    - Imagen 128x128 (75% menos memoria que 256x256)
    - Arquitectura lite (filtros reducidos)
    - Limpieza de memoria cada epoch
    
Loss y Métricas:
    - Loss combinada: Binary CE + Dice Loss
    - IoU (Intersection over Union)
    - Dice Coefficient
    - Pixel Accuracy

Autor: Sistema de Detección de Fisuras
Fecha: Octubre 2025
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

# Scikit-learn
from sklearn.metrics import confusion_matrix, classification_report

# Configuración del proyecto
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    RUTA_SEGMENTACION,
    RUTA_MODELO_SEGMENTACION,
    RUTA_VIS,
    RANDOM_SEED
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Directorios
DATOS_DIR = Path(RUTA_SEGMENTACION)
MODELOS_DIR = Path(RUTA_MODELO_SEGMENTACION)
VIS_DIR = Path(RUTA_VIS)

MODELOS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de imagen
IMG_SIZE = 128  # Reducido de 256 a 128 para ahorrar VRAM (75% menos memoria)
IMG_CHANNELS = 3

# Parámetros de entrenamiento
BATCH_SIZE = 4  # Muy reducido para RTX 2050 (1.7GB VRAM disponible)
EPOCHS = 50
LEARNING_RATE = 1e-4

# Semilla para reproducibilidad
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# OPTIMIZACIONES GPU
# ============================================================================

print("🔧 Configurando optimizaciones GPU...")

# Mixed Precision para RTX 2050
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
print(f"   ✅ Mixed Precision: {policy.name}")

# XLA JIT Compilation
tf.config.optimizer.set_jit(True)
print("   ✅ XLA JIT Compilation habilitado")

# Configurar GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   ✅ GPU detectada: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        print(f"   ⚠️  Error configurando GPU: {e}")
else:
    print("   ⚠️  No se detectó GPU, usando CPU")


# ============================================================================
# CALLBACK PERSONALIZADO: LIMPIEZA DE MEMORIA
# ============================================================================

class MemoryCleanupCallback(keras.callbacks.Callback):
    """
    Callback para limpiar memoria GPU al final de cada epoch.
    Ayuda a prevenir fragmentación de memoria en GPUs pequeñas.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Limpia cache de GPU al finalizar epoch"""
        import gc
        gc.collect()
        keras.backend.clear_session()
        # Re-aplicar mixed precision después de clear_session
        keras.mixed_precision.set_global_policy('mixed_float16')


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU) / Jaccard Index.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice Coefficient (F1-Score para segmentación).
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice Loss = 1 - Dice Coefficient.
    Útil para clases desbalanceadas.
    """
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """
    Loss combinada: Binary Crossentropy + Dice Loss.
    
    BCE se enfoca en píxeles individuales.
    Dice Loss se enfoca en la superposición global.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


# ============================================================================
# ARQUITECTURA U-NET
# ============================================================================

def conv_block(inputs, filters, kernel_size=3):
    """
    Bloque convolucional básico: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU
    """
    x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x


def encoder_block(inputs, filters):
    """
    Bloque encoder: Conv Block → Max Pooling
    Retorna features para skip connection y pooled output
    """
    conv = conv_block(inputs, filters)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def decoder_block(inputs, skip_features, filters):
    """
    Bloque decoder: UpSampling → Concatenate con skip → Conv Block
    """
    up = layers.UpSampling2D(size=(2, 2))(inputs)
    concat = layers.Concatenate()([up, skip_features])
    conv = conv_block(concat, filters)
    return conv


def build_unet(input_shape=(128, 128, 3)):
    """
    Construye arquitectura U-Net OPTIMIZADA para RTX 2050 (1.7GB VRAM).
    
    Cambios vs U-Net estándar:
    - Filtros reducidos: 32→64→128→256 (en vez de 64→128→256→512→1024)
    - Imagen 128x128 (en vez de 256x256) = 75% menos memoria
    - Sin bottleneck de 1024 filtros (máximo 256)
    - Parámetros: ~2M (en vez de 31M) = 93% reducción
    
    Args:
        input_shape: Forma de entrada (height, width, channels)
    
    Returns:
        Modelo de Keras optimizado
    """
    inputs = layers.Input(input_shape)
    
    # Encoder (contracting path) - FILTROS REDUCIDOS
    skip1, pool1 = encoder_block(inputs, 32)       # 128x128 → 64x64
    skip2, pool2 = encoder_block(pool1, 64)        # 64x64 → 32x32
    skip3, pool3 = encoder_block(pool2, 128)       # 32x32 → 16x16
    
    # Bottleneck (reducido de 1024 a 256)
    bottleneck = conv_block(pool3, 256)            # 16x16
    
    # Decoder (expanding path) con skip connections
    dec3 = decoder_block(bottleneck, skip3, 128)   # 16x16 → 32x32
    dec2 = decoder_block(dec3, skip2, 64)          # 32x32 → 64x64
    dec1 = decoder_block(dec2, skip1, 32)          # 64x64 → 128x128
    
    # Output layer (máscara binaria)
    outputs = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32')(dec1)
    
    model = Model(inputs=inputs, outputs=outputs, name='U-Net-Lite')
    return model


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def combinar_generadores(image_gen, mask_gen):
    """
    Combina generadores de imágenes y máscaras en un solo generador.
    
    Args:
        image_gen: Generador de imágenes
        mask_gen: Generador de máscaras
    
    Yields:
        tuple: (batch_images, batch_masks)
    """
    while True:
        img_batch = next(image_gen)
        mask_batch = next(mask_gen)
        yield (img_batch, mask_batch)


def crear_generadores(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Crea generadores de datos con augmentation para train y sin augmentation para val/test.
    
    Returns:
        tuple: (train_gen, val_gen, test_gen)
    """
    print("\n📦 Creando generadores de datos...")
    
    # Data augmentation para training
    train_image_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    train_mask_datagen = ImageDataGenerator(
        rescale=1./255,  # CORREGIDO: bool se convierte a 0.0/1.0 correctamente
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    # Sin augmentation para validation y test
    val_test_image_datagen = ImageDataGenerator(rescale=1./255)
    val_test_mask_datagen = ImageDataGenerator(rescale=1./255)  # CORREGIDO
    
    # Configuración común
    flow_params = {
        'target_size': (img_size, img_size),
        'batch_size': batch_size,
        'class_mode': None,
        'seed': RANDOM_SEED
    }
    
    # Train generators
    train_image_gen = train_image_datagen.flow_from_directory(
        DATOS_DIR / 'train',
        classes=['images'],
        color_mode='rgb',
        **flow_params
    )
    
    train_mask_gen = train_mask_datagen.flow_from_directory(
        DATOS_DIR / 'train',
        classes=['masks'],
        color_mode='grayscale',
        **flow_params
    )
    
    # Val generators
    val_image_gen = val_test_image_datagen.flow_from_directory(
        DATOS_DIR / 'val',
        classes=['images'],
        color_mode='rgb',
        **flow_params
    )
    
    val_mask_gen = val_test_mask_datagen.flow_from_directory(
        DATOS_DIR / 'val',
        classes=['masks'],
        color_mode='grayscale',
        **flow_params
    )
    
    # Test generators
    test_image_gen = val_test_image_datagen.flow_from_directory(
        DATOS_DIR / 'test',
        classes=['images'],
        color_mode='rgb',
        **flow_params
    )
    
    test_mask_gen = val_test_mask_datagen.flow_from_directory(
        DATOS_DIR / 'test',
        classes=['masks'],
        color_mode='grayscale',
        **flow_params
    )
    
    # Combinar imágenes y máscaras con función generadora
    train_gen = combinar_generadores(train_image_gen, train_mask_gen)
    val_gen = combinar_generadores(val_image_gen, val_mask_gen)
    test_gen = combinar_generadores(test_image_gen, test_mask_gen)
    
    print(f"   ✅ Train: {train_image_gen.n} imágenes")
    print(f"   ✅ Val:   {val_image_gen.n} imágenes")
    print(f"   ✅ Test:  {test_image_gen.n} imágenes")
    
    return train_gen, val_gen, test_gen, train_image_gen.n, val_image_gen.n, test_image_gen.n


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_modelo():
    """
    Función principal de entrenamiento.
    """
    print("\n" + "="*70)
    print("🏗️  ENTRENAMIENTO DE U-NET PARA SEGMENTACIÓN DE FISURAS")
    print("="*70)
    
    # 1. Crear generadores
    train_gen, val_gen, test_gen, n_train, n_val, n_test = crear_generadores()
    
    steps_per_epoch = n_train // BATCH_SIZE
    validation_steps = n_val // BATCH_SIZE
    
    # 2. Construir modelo
    print("\n🏗️  Construyendo arquitectura U-Net...")
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    
    # Resumen del modelo
    total_params = model.count_params()
    print(f"   ✅ Parámetros totales: {total_params:,}")
    print(f"   ✅ Tamaño estimado: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # 3. Compilar modelo
    print("\n⚙️  Compilando modelo...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[
            'accuracy',
            iou_metric,
            dice_coefficient
        ]
    )
    print("   ✅ Loss: Binary CE + Dice Loss")
    print("   ✅ Métricas: Accuracy, IoU, Dice")
    
    # 4. Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=MODELOS_DIR / f'best_unet_{timestamp}.keras',
            monitor='val_iou_metric',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODELOS_DIR / 'unet_segmentacion_final.keras',
            monitor='val_iou_metric',
            mode='max',
            save_best_only=True,
            verbose=0
        ),
        EarlyStopping(
            monitor='val_iou_metric',
            patience=15,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # NO USAR MemoryCleanupCallback - causa problemas con el modelo compilado
        # La limpieza automática de TF con memory_growth=True es suficiente
    ]
    
    # 5. Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    print("\n" + "-"*70)
    
    inicio = time.time()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    tiempo_total = time.time() - inicio
    
    print("\n" + "-"*70)
    print(f"✅ Entrenamiento completado en {tiempo_total/60:.2f} minutos")
    
    # 6. Guardar historial
    history_path = MODELOS_DIR / f'history_unet_{timestamp}.json'
    with open(history_path, 'w') as f:
        # Convertir arrays numpy a listas para JSON
        history_dict = {key: [float(val) for val in values] 
                       for key, values in history.history.items()}
        history_dict['tiempo_total_min'] = tiempo_total / 60
        json.dump(history_dict, f, indent=2)
    
    print(f"   📄 Historial guardado: {history_path}")
    
    # 7. Visualizar curvas
    visualizar_entrenamiento(history, timestamp)
    
    # 8. Evaluar en test set
    print("\n📊 Evaluando en test set...")
    _, _, test_gen_new, _, _, n_test = crear_generadores()
    test_steps = n_test // BATCH_SIZE
    
    test_results = model.evaluate(
        test_gen_new,
        steps=test_steps,
        verbose=1
    )
    
    print("\n🎯 Resultados en Test Set:")
    print(f"   Loss: {test_results[0]:.4f}")
    print(f"   Accuracy: {test_results[1]*100:.2f}%")
    print(f"   IoU: {test_results[2]:.4f}")
    print(f"   Dice: {test_results[3]:.4f}")
    
    # Guardar resultados
    resultados = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_iou': float(test_results[2]),
        'test_dice': float(test_results[3]),
        'timestamp': timestamp,
        'tiempo_entrenamiento_min': tiempo_total / 60,
        'epochs_entrenados': len(history.history['loss']),
        'mejor_val_iou': float(max(history.history['val_iou_metric']))
    }
    
    with open(MODELOS_DIR / f'resultados_test_{timestamp}.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
    print("="*70)
    print(f"\n📁 Modelos guardados en: {MODELOS_DIR}")
    print(f"📊 Visualizaciones en: {VIS_DIR}")
    
    return model, history


def visualizar_entrenamiento(history, timestamp):
    """
    Genera gráficos de las curvas de entrenamiento.
    """
    print("\n📊 Generando visualizaciones...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss (Binary CE + Dice)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # IoU
    axes[0, 1].plot(history.history['iou_metric'], label='Train IoU', linewidth=2)
    axes[0, 1].plot(history.history['val_iou_metric'], label='Val IoU', linewidth=2)
    axes[0, 1].set_title('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Dice Coefficient
    axes[1, 0].plot(history.history['dice_coefficient'], label='Train Dice', linewidth=2)
    axes[1, 0].plot(history.history['val_dice_coefficient'], label='Val Dice', linewidth=2)
    axes[1, 0].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Accuracy
    axes[1, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1, 1].set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    save_path = VIS_DIR / f'training_curves_unet_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Curvas guardadas: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    model, history = entrenar_modelo()
