"""
🤖 Modelo CNN Simplificado para Detección de Fisuras
Versión simple y robusta que funciona garantizado
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class SimpleCrackDetector:
    def __init__(self, 
                 data_path="data/processed/sdnet2018_prepared",
                 img_size=(128, 128),  # Más pequeño para ser más rápido
                 batch_size=32):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Crear directorios
        Path("models").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
    def load_data(self):
        """Cargar datos usando tf.keras.utils.image_dataset_from_directory"""
        print("📊 CARGANDO DATOS")
        print("=" * 50)
        
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
        
        # Normalizar datos
        normalization_layer = layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
        
        print(f"✅ Datos cargados y optimizados")
        return train_ds, val_ds, test_ds
    
    def build_model(self):
        """Construir modelo CNN simple"""
        print(f"\n🏗️ CONSTRUYENDO MODELO CNN SIMPLE")
        print("=" * 50)
        
        self.model = models.Sequential([
            # Entrada
            layers.Input(shape=(*self.img_size, 3)),
            
            # Bloque 1
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Bloque 2
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Bloque 3
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Clasificador
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo construido: {self.model.count_params():,} parámetros")
        return self.model
    
    def train_model(self, train_ds, val_ds, epochs=15):
        """Entrenar modelo"""
        print(f"\n🚀 ENTRENANDO MODELO ({epochs} épocas)")
        print("=" * 50)
        
        # Callbacks simples
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        checkpoint = keras.callbacks.ModelCheckpoint(
            'models/simple_crack_detector.keras',
            save_best_only=True,
            monitor='val_accuracy'
        )
        
        # Entrenar
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        print(f"✅ Entrenamiento completado!")
        return self.history
    
    def evaluate_model(self, test_ds):
        """Evaluar modelo"""
        print(f"\n📊 EVALUANDO MODELO")
        print("=" * 50)
        
        # Cargar mejor modelo
        try:
            self.model = keras.models.load_model('models/simple_crack_detector.keras')
            print("✅ Modelo cargado desde checkpoint")
        except:
            print("⚠️ Usando modelo actual")
        
        # Evaluación
        test_loss, test_accuracy = self.model.evaluate(test_ds, verbose=0)
        
        print(f"📈 RESULTADOS:")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Test Loss:     {test_loss:.4f}")
        
        # Guardar métricas
        metrics = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss)
        }
        
        with open('results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def plot_history(self):
        """Graficar historial de entrenamiento"""
        if self.history is None:
            return
        
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Gráficos guardados en results/training_history.png")
    
    def predict_sample(self, test_ds, num_samples=6):
        """Hacer predicciones de muestra"""
        print(f"\n🔍 PREDICCIONES DE MUESTRA")
        print("=" * 50)
        
        # Obtener un batch de prueba
        for images, labels in test_ds.take(1):
            predictions = self.model.predict(images[:num_samples])
            
            plt.figure(figsize=(15, 10))
            for i in range(num_samples):
                plt.subplot(2, 3, i + 1)
                plt.imshow(images[i].numpy())
                
                true_label = "Crack" if labels[i] == 1 else "No Crack"
                pred_prob = predictions[i][0]
                pred_label = "Crack" if pred_prob > 0.5 else "No Crack"
                confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
                
                color = 'green' if true_label == pred_label else 'red'
                plt.title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})', 
                         color=color)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('results/sample_predictions.png', dpi=150, bbox_inches='tight')
            plt.show()
            break
        
        print("🖼️ Predicciones guardadas en results/sample_predictions.png")
    
    def run_complete_training(self):
        """Ejecutar entrenamiento completo"""
        print("🎯 ENTRENAMIENTO COMPLETO - DETECTOR SIMPLE")
        print("=" * 60)
        
        # 1. Cargar datos
        train_ds, val_ds, test_ds = self.load_data()
        
        # 2. Construir modelo
        self.build_model()
        
        # 3. Entrenar
        self.train_model(train_ds, val_ds, epochs=10)  # Pocas épocas para ser rápido
        
        # 4. Evaluar
        metrics = self.evaluate_model(test_ds)
        
        # 5. Visualizar
        self.plot_history()
        self.predict_sample(test_ds)
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
        print(f"📊 Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"💾 Modelo guardado en: models/simple_crack_detector.keras")
        
        return metrics

if __name__ == "__main__":
    # Ejecutar entrenamiento
    detector = SimpleCrackDetector()
    results = detector.run_complete_training()
    
    print(f"\n🏆 ¡MODELO LISTO!")
    print(f"✅ Accuracy: {results['test_accuracy']:.2%}")
    print(f"📁 Archivos en: models/ y results/")