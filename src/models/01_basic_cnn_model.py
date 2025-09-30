"""
🤖 Modelo CNN Básico para Detección de Fisuras
Modelo simple pero efectivo usando Transfer Learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import image_dataset_from_directory

from sklearn.metrics import classification_report, confusion_matrix
import json

class CrackDetectionModel:
    def __init__(self, 
                 data_path="data/processed/sdnet2018_prepared",
                 img_size=(224, 224),
                 batch_size=32):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Crear directorio para modelos
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Crear directorio para resultados
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def create_data_generators(self):
        """Crear generadores de datos con augmentation"""
        print("📊 CONFIGURANDO GENERADORES DE DATOS")
        print("=" * 50)
        
        # Data Augmentation para entrenamiento
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # Solo normalización para validación y test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Generador de entrenamiento
        train_generator = train_datagen.flow_from_directory(
            self.data_path / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        # Generador de validación
        validation_generator = val_test_datagen.flow_from_directory(
            self.data_path / 'validation',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            seed=42
        )
        
        # Generador de prueba
        test_generator = val_test_datagen.flow_from_directory(
            self.data_path / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            seed=42
        )
        
        print(f"✅ Entrenamiento: {train_generator.samples} imágenes")
        print(f"✅ Validación: {validation_generator.samples} imágenes")
        print(f"✅ Prueba: {test_generator.samples} imágenes")
        print(f"📋 Clases: {list(train_generator.class_indices.keys())}")
        
        return train_generator, validation_generator, test_generator
    
    def build_model(self):
        """Construir modelo CNN con Transfer Learning"""
        print(f"\n🏗️ CONSTRUYENDO MODELO CNN")
        print("=" * 50)
        
        # Modelo CNN simple pero efectivo
        self.model = models.Sequential([
            # Capas convolucionales
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Capas densas
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Clasificación binaria
        ])
        
        # Compilar modelo
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Modelo construido con {self.model.count_params():,} parámetros")
        print(f"📊 Arquitectura: EfficientNetB0 + Custom Head")
        
        return self.model
    
    def setup_callbacks(self):
        """Configurar callbacks para entrenamiento"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                self.models_dir / 'best_crack_detection_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, train_gen, val_gen, epochs=30):
        """Entrenar modelo"""
        print(f"\n🚀 INICIANDO ENTRENAMIENTO ({epochs} épocas)")
        print("=" * 50)
        
        callbacks = self.setup_callbacks()
        
        # Entrenar modelo
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✅ Entrenamiento completado!")
        return self.history
    
    def evaluate_model(self, test_gen):
        """Evaluar modelo con conjunto de prueba"""
        print(f"\n📊 EVALUANDO MODELO")
        print("=" * 50)
        
        # Cargar mejor modelo
        best_model_path = self.models_dir / 'best_crack_detection_model.keras'
        if best_model_path.exists():
            self.model = tf.keras.models.load_model(best_model_path)
            print(f"✅ Modelo cargado desde: {best_model_path}")
        
        # Evaluación básica
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_gen, verbose=0
        )
        
        # Predicciones detalladas
        predictions = self.model.predict(test_gen, verbose=0)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        
        # Cálculos adicionales
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        # Métricas detalladas
        print(f"📈 MÉTRICAS DE EVALUACIÓN:")
        print(f"   Accuracy:  {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1-Score:  {f1_score:.4f}")
        print(f"   Loss:      {test_loss:.4f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm)
        
        # Reporte detallado
        class_names = ['No Crack', 'Crack']
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Guardar métricas
        metrics = {
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1_score': float(f1_score),
            'test_loss': float(test_loss),
            'classification_report': report
        }
        
        with open(self.results_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"📄 Métricas guardadas en: {self.results_dir / 'evaluation_metrics.json'}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm):
        """Visualizar matriz de confusión"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Crack', 'Crack'],
                   yticklabels=['No Crack', 'Crack'])
        plt.title('Matriz de Confusión - Detección de Fisuras')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Matriz de confusión guardada: {self.results_dir / 'confusion_matrix.png'}")
    
    def plot_training_history(self):
        """Visualizar historial de entrenamiento"""
        if self.history is None:
            print("⚠️ No hay historial de entrenamiento disponible")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0,0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0,0].set_title('Accuracy durante Entrenamiento')
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Loss
        axes[0,1].plot(self.history.history['loss'], label='Train Loss')
        axes[0,1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0,1].set_title('Loss durante Entrenamiento')
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Precision
        axes[1,0].plot(self.history.history['precision'], label='Train Precision')
        axes[1,0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1,0].set_title('Precision durante Entrenamiento')
        axes[1,0].set_xlabel('Época')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Recall
        axes[1,1].plot(self.history.history['recall'], label='Train Recall')
        axes[1,1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1,1].set_title('Recall durante Entrenamiento')
        axes[1,1].set_xlabel('Época')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Historial guardado: {self.results_dir / 'training_history.png'}")
    
    def train_complete_pipeline(self, epochs=25):
        """Pipeline completo de entrenamiento"""
        print("🎯 PIPELINE COMPLETO DE ENTRENAMIENTO")
        print("=" * 60)
        
        # 1. Crear generadores de datos
        train_gen, val_gen, test_gen = self.create_data_generators()
        
        # 2. Construir modelo
        self.build_model()
        
        # 3. Entrenar modelo
        self.train_model(train_gen, val_gen, epochs=epochs)
        
        # 4. Visualizar entrenamiento
        self.plot_training_history()
        
        # 5. Evaluar modelo
        metrics = self.evaluate_model(test_gen)
        
        print(f"\n🎉 PIPELINE COMPLETADO")
        print(f"📊 Accuracy final: {metrics['test_accuracy']:.4f}")
        print(f"📁 Modelo guardado en: {self.models_dir}")
        print(f"📈 Resultados en: {self.results_dir}")
        
        return metrics

if __name__ == "__main__":
    # Ejecutar entrenamiento completo
    detector = CrackDetectionModel()
    final_metrics = detector.train_complete_pipeline(epochs=20)
    
    print(f"\n🏆 MODELO ENTRENADO EXITOSAMENTE")
    print(f"✅ Listo para clasificación de severidad!")