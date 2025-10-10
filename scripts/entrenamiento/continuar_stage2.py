"""
SCRIPT PARA CONTINUAR STAGE 2 DESDE MODELO GUARDADO
====================================================

Este script carga el modelo guardado de Stage 1 y continúa con el
fine-tuning completo (Stage 2) usando batch size reducido para evitar OOM.

Uso:
    cd /mnt/c/Users/jonna/OneDrive/Escritorio/DEEP\ LEARNING/investigacion_fisuras
    source venv/bin/activate
    python3 scripts/entrenamiento/continuar_stage2.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Configurar GPU ANTES de importar TensorFlow
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.utils.configurar_gpu import configurar_gpu_maxima_velocidad
config_gpu = configurar_gpu_maxima_velocidad(verbose=True)

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from config import (
    RUTA_DETECCION,
    RUTA_MODELO_DETECCION,
    RUTA_VIS,
    IMG_SIZE,
    RANDOM_SEED,
    AUGMENTATION_CONFIG
)

# Configuración
DATOS_PROCESADOS_DETECCION = Path(RUTA_DETECCION)
MODELOS_DETECCION = Path(RUTA_MODELO_DETECCION)
RESULTADOS_VISUALIZACIONES = Path(RUTA_VIS)

# Parámetros Stage 2
BATCH_SIZE_STAGE2 = 48  # Reducido de 64 para evitar OOM
EPOCHS_STAGE2 = 22
LEARNING_RATE_STAGE2 = 1e-4

print("\n" + "=" * 80)
print("CONTINUACIÓN DE ENTRENAMIENTO - STAGE 2 SOLAMENTE")
print("=" * 80)
print(f"Batch size Stage 2: {BATCH_SIZE_STAGE2} (reducido para evitar OOM)")
print(f"Epochs Stage 2: {EPOCHS_STAGE2}")
print(f"Learning rate Stage 2: {LEARNING_RATE_STAGE2}")
print("=" * 80 + "\n")


def calcular_class_weights(train_dir):
    """Calcula class weights"""
    import glob
    
    cracked = len(glob.glob(str(train_dir / 'cracked' / '*')))
    uncracked = len(glob.glob(str(train_dir / 'uncracked' / '*')))
    total = cracked + uncracked
    
    weight_cracked = total / (2 * cracked)
    weight_uncracked = total / (2 * uncracked)
    
    class_weights = {
        0: weight_uncracked,
        1: weight_cracked
    }
    
    print(f"📊 Class weights:")
    print(f"   Cracked: {weight_cracked:.3f}")
    print(f"   Uncracked: {weight_uncracked:.3f}\n")
    
    return class_weights


def crear_generadores_stage2(train_dir, val_dir):
    """Crea generadores con batch size reducido"""
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        vertical_flip=AUGMENTATION_CONFIG['vertical_flip'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        brightness_range=AUGMENTATION_CONFIG['brightness_range'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode']
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE_STAGE2,
        class_mode='binary',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE_STAGE2,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"📦 Generadores creados:")
    print(f"   Train samples: {train_generator.samples}")
    print(f"   Val samples: {val_generator.samples}")
    print(f"   Batches por época: {len(train_generator)}\n")
    
    return train_generator, val_generator


def crear_callbacks_stage2():
    """Crea callbacks para Stage 2"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(MODELOS_DETECCION / 'best_model_stage2.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        TensorBoard(
            log_dir=str(MODELOS_DETECCION / 'logs' / f'stage2_{timestamp}'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        )
    ]
    
    return callbacks


def main():
    """Función principal"""
    
    inicio_total = time.time()
    
    # Verificar datos
    train_dir = DATOS_PROCESADOS_DETECCION / 'train'
    val_dir = DATOS_PROCESADOS_DETECCION / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Datos no encontrados en {DATOS_PROCESADOS_DETECCION}")
    
    # Calcular class weights
    class_weights = calcular_class_weights(train_dir)
    
    # Crear generadores
    train_gen, val_gen = crear_generadores_stage2(train_dir, val_dir)
    
    # Cargar modelo de Stage 1
    print("=" * 80)
    print("CARGANDO MODELO DE STAGE 1")
    print("=" * 80)
    
    modelo_stage1_path = MODELOS_DETECCION / 'best_model_stage1.keras'
    
    if not modelo_stage1_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de Stage 1 en: {modelo_stage1_path}\n"
            "Ejecuta primero el entrenamiento de Stage 1."
        )
    
    print(f"Cargando desde: {modelo_stage1_path}")
    model = tf.keras.models.load_model(modelo_stage1_path)
    print("✅ Modelo Stage 1 cargado exitosamente\n")
    
    # Obtener base model (MobileNetV2)
    base_model = None
    for layer in model.layers:
        if 'mobilenetv2' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        raise ValueError("No se encontró la capa MobileNetV2 en el modelo")
    
    # Descongelar base model
    print("=" * 80)
    print("STAGE 2: FINE-TUNING COMPLETO")
    print("=" * 80)
    print("🔓 Descongelando base MobileNetV2...")
    
    base_model.trainable = True
    
    # Recompilar con learning rate bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ],
        jit_compile=True
    )
    
    trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    print(f"✅ Modelo recompilado para fine-tuning")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Batch size: {BATCH_SIZE_STAGE2}")
    print(f"   Learning rate: {LEARNING_RATE_STAGE2}\n")
    
    # Crear callbacks
    callbacks = crear_callbacks_stage2()
    
    # Entrenar Stage 2
    print("🚀 Iniciando Stage 2...")
    print("=" * 80 + "\n")
    
    inicio_stage2 = time.time()
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS_STAGE2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    tiempo_stage2 = time.time() - inicio_stage2
    tiempo_total = time.time() - inicio_total
    
    # Guardar modelo final
    print("\n" + "=" * 80)
    print("💾 GUARDANDO MODELO FINAL")
    print("=" * 80)
    
    modelo_final_path = MODELOS_DETECCION / 'modelo_deteccion_final.keras'
    model.save(modelo_final_path)
    print(f"✅ Modelo final guardado en: {modelo_final_path}")
    
    # Resumen
    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETADO")
    print("=" * 80)
    print(f"⏱️ Tiempo Stage 2: {tiempo_stage2/60:.2f} minutos")
    print(f"⏱️ Tiempo total: {tiempo_total/60:.2f} minutos")
    print(f"📊 Mejor val_auc: {max(history.history['val_auc']):.4f}")
    print(f"📊 Mejor val_accuracy: {max(history.history['val_accuracy']):.4f}")
    print("=" * 80)
    
    print("\n✅ ¡Entrenamiento completado exitosamente!")
    print(f"   Modelo final: {modelo_final_path}")
    print(f"   Mejor Stage 2: {MODELOS_DETECCION / 'best_model_stage2.keras'}")


if __name__ == '__main__':
    main()
