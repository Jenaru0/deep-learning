"""
Modelo CNN Mejorado con Dataset Unificado
==========================================

Re-entrenamiento del modelo CNN con el dataset unificado SDNET2018 + CRACK500
para obtener mejor precisión y robustez.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedCrackDetector:
    """Modelo CNN mejorado entrenado con dataset unificado"""
    
    def __init__(self, 
                 data_path="data/processed/unified_dataset",
                 img_size=(128, 128),
                 batch_size=32,
                 epochs=15):  # Más épocas por dataset más grande
        
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None
        
        # Crear directorios de resultados
        Path("models").mkdir(exist_ok=True)
        Path("results/improved_model").mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Cargar dataset unificado"""
        print("📊 CARGANDO DATASET UNIFICADO")
        print("=" * 50)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset unificado no encontrado: {self.data_path}")
        
        # Dataset de entrenamiento
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path / 'train',
            image_size=self.img_size,
            batch_size=self.batch_size,
            seed=42
        )
        
        # Dataset de validación
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path / 'validation',
            image_size=self.img_size,
            batch_size=self.batch_size,
            seed=42
        )
        
        # Dataset de prueba
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path / 'test',
            image_size=self.img_size,
            batch_size=self.batch_size,
            seed=42
        )
        
        # Optimizar rendimiento
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        print("✅ Dataset unificado cargado y optimizado")
        
        return train_ds, val_ds, test_ds
    
    def build_improved_model(self):
        """Construir modelo CNN mejorado"""
        print("🏗️ CONSTRUYENDO MODELO CNN MEJORADO")
        print("=" * 50)
        
        model = keras.Sequential([
            # Capa de entrada
            layers.Input(shape=(*self.img_size, 3)),
            
            # Normalización
            layers.Rescaling(1./255),
            
            # Aumento de datos
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Bloque convolucional 1
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Bloque convolucional 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Bloque convolucional 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            
            # Bloque convolucional 4 (nuevo)
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Capas densas
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Mostrar resumen
        print(f"✅ Modelo mejorado construido: {model.count_params():,} parámetros")
        return model
    
    def train_model(self, train_ds, val_ds):
        """Entrenar el modelo mejorado"""
        print("🚀 ENTRENANDO MODELO MEJORADO")
        print("=" * 50)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'models/improved_crack_detector_best.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Entrenar
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks
        )
        
        print("✅ Entrenamiento completado!")
        return self.history
    
    def evaluate_model(self, test_ds):
        """Evaluar modelo mejorado"""
        print("📊 EVALUANDO MODELO MEJORADO")
        print("=" * 50)
        
        # Cargar mejor modelo
        best_model = keras.models.load_model('models/improved_crack_detector_best.keras')
        
        # Evaluar
        test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
        
        print(f"📈 RESULTADOS MODELO MEJORADO:")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss:     {test_loss:.4f}")
        
        # Guardar métricas
        metrics = {
            'model_type': 'improved_cnn_unified_dataset',
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'dataset': 'SDNET2018_CRACK500_Unified',
            'total_params': int(best_model.count_params()),
            'training_date': datetime.now().isoformat(),
            'epochs_trained': len(self.history.history['accuracy']),
            'best_val_accuracy': float(max(self.history.history['val_accuracy']))
        }
        
        with open('results/improved_model/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return test_acc, test_loss, metrics
    
    def plot_training_history(self):
        """Graficar historial de entrenamiento"""
        print("📊 GENERANDO GRÁFICOS DE ENTRENAMIENTO")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax1.set_title('Modelo Mejorado - Accuracy')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss', color='blue')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        ax2.set_title('Modelo Mejorado - Loss')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/improved_model/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Gráficos guardados en results/improved_model/training_history.png")
    
    def create_sample_predictions(self, test_ds):
        """Crear predicciones de muestra"""
        print("🔍 GENERANDO PREDICCIONES DE MUESTRA")
        
        # Cargar mejor modelo
        best_model = keras.models.load_model('models/improved_crack_detector_best.keras')
        
        # Obtener muestra de datos
        images, labels = next(iter(test_ds))
        predictions = best_model.predict(images[:16])
        
        # Crear visualización
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Predicciones del Modelo Mejorado', fontsize=16)
        
        class_names = ['No Crack', 'Crack']
        
        for i in range(16):
            ax = axes[i//4, i%4]
            
            # Mostrar imagen
            ax.imshow(images[i].numpy().astype("uint8"))
            
            # Título con predicción
            predicted_class = int(predictions[i] > 0.5)
            actual_class = int(labels[i])
            confidence = predictions[i][0] if predicted_class == 1 else 1 - predictions[i][0]
            
            color = 'green' if predicted_class == actual_class else 'red'
            title = f'Real: {class_names[actual_class]}\\nPred: {class_names[predicted_class]} ({confidence:.2f})'
            ax.set_title(title, color=color, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/improved_model/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Predicciones guardadas en results/improved_model/sample_predictions.png")
    
    def train_complete_pipeline(self):
        """Pipeline completo de entrenamiento"""
        print("🎯 ENTRENAMIENTO COMPLETO - MODELO MEJORADO")
        print("=" * 60)
        
        # Cargar datos
        train_ds, val_ds, test_ds = self.load_data()
        
        # Construir modelo
        self.build_improved_model()
        
        # Entrenar
        self.train_model(train_ds, val_ds)
        
        # Evaluar
        test_acc, test_loss, metrics = self.evaluate_model(test_ds)
        
        # Visualizaciones
        self.plot_training_history()
        self.create_sample_predictions(test_ds)
        
        # Guardar modelo final
        self.model.save('models/improved_crack_detector_final.keras')
        
        print("\n" + "=" * 60)
        print("🎉 ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"🎯 Accuracy final: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"💾 Modelo guardado: models/improved_crack_detector_best.keras")
        print(f"📊 Métricas: results/improved_model/metrics.json")
        
        return test_acc, metrics

if __name__ == "__main__":
    # Crear y entrenar modelo mejorado
    detector = ImprovedCrackDetector()
    final_accuracy, metrics = detector.train_complete_pipeline()
    
    print(f"\n🏆 ¡MODELO MEJORADO LISTO!")
    print(f"✅ Accuracy: {final_accuracy*100:.2f}%")
    print(f"📁 Archivos en: models/ y results/improved_model/")