"""
Script de Preprocesamiento: División Estratificada de SDNET2018
================================================================
Divide el dataset SDNET2018 en train/val/test (70/15/15) con estratificación
por categoría (Deck/Pavement/Wall) y clase (Cracked/Uncracked).

Autor: Sistema de Detección de Fisuras
Fecha: 7 de octubre, 2025
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Agregar la ruta del proyecto al sys.path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Importar sklearn para división estratificada
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("⚠️  scikit-learn no está instalado. Instalando...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RANDOM_SEED = config.RANDOM_SEED
TRAIN_RATIO = config.TRAIN_RATIO
VAL_RATIO = config.VAL_RATIO
TEST_RATIO = config.TEST_RATIO

# Rutas
RUTA_SDNET2018 = Path(config.RUTA_SDNET2018)
RUTA_DETECCION = Path(config.RUTA_DETECCION)

# Mapeo de categorías y clases
CATEGORIAS = {
    'D': {'dir': 'D', 'nombre': 'Deck', 'cracked': 'CD', 'uncracked': 'UD'},
    'P': {'dir': 'P', 'nombre': 'Pavement', 'cracked': 'CP', 'uncracked': 'UP'},
    'W': {'dir': 'W', 'nombre': 'Wall', 'cracked': 'CW', 'uncracked': 'UW'}
}

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def recolectar_imagenes():
    """
    Recolecta todas las imágenes de SDNET2018 organizadas por categoría y clase.
    
    Returns:
        dict: Diccionario con estructura {categoria: {clase: [lista_rutas]}}
    """
    print("📂 Paso 1/5: Recolectando imágenes de SDNET2018...")
    
    imagenes = defaultdict(lambda: defaultdict(list))
    total_encontradas = 0
    
    for cat_key, cat_info in CATEGORIAS.items():
        cat_dir = RUTA_SDNET2018 / cat_info['dir']
        
        # Procesar imágenes con grietas (cracked)
        cracked_dir = cat_dir / cat_info['cracked']
        if cracked_dir.exists():
            cracked_imgs = list(cracked_dir.glob("*.jpg"))
            imagenes[cat_key]['cracked'].extend(cracked_imgs)
            print(f"  ✓ {cat_info['nombre']}/Cracked: {len(cracked_imgs)} imágenes")
            total_encontradas += len(cracked_imgs)
        
        # Procesar imágenes sin grietas (uncracked)
        uncracked_dir = cat_dir / cat_info['uncracked']
        if uncracked_dir.exists():
            uncracked_imgs = list(uncracked_dir.glob("*.jpg"))
            imagenes[cat_key]['uncracked'].extend(uncracked_imgs)
            print(f"  ✓ {cat_info['nombre']}/Uncracked: {len(uncracked_imgs)} imágenes")
            total_encontradas += len(uncracked_imgs)
    
    print(f"\n📊 Total de imágenes encontradas: {total_encontradas}")
    return imagenes


def dividir_estratificado(imagenes):
    """
    Divide las imágenes en train/val/test de forma estratificada.
    Mantiene proporción de categorías y clases en cada split.
    
    Args:
        imagenes (dict): Diccionario con imágenes organizadas
        
    Returns:
        dict: Splits organizados {train: {cracked: [], uncracked: []}, ...}
    """
    print(f"\n🔀 Paso 2/5: Dividiendo dataset ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")
    
    # Fijar semilla para reproducibilidad
    random.seed(RANDOM_SEED)
    
    splits = {
        'train': {'cracked': [], 'uncracked': []},
        'val': {'cracked': [], 'uncracked': []},
        'test': {'cracked': [], 'uncracked': []}
    }
    
    # Dividir por categoría y clase para mantener estratificación
    for cat_key, cat_info in CATEGORIAS.items():
        for clase in ['cracked', 'uncracked']:
            imgs = imagenes[cat_key][clase]
            
            if len(imgs) == 0:
                continue
            
            # Primera división: train vs (val+test)
            train_imgs, temp_imgs = train_test_split(
                imgs,
                train_size=TRAIN_RATIO,
                random_state=RANDOM_SEED,
                shuffle=True
            )
            
            # Segunda división: val vs test (del 30% restante)
            # VAL_RATIO / (VAL_RATIO + TEST_RATIO) = 0.15 / 0.30 = 0.5
            val_imgs, test_imgs = train_test_split(
                temp_imgs,
                train_size=0.5,
                random_state=RANDOM_SEED,
                shuffle=True
            )
            
            splits['train'][clase].extend(train_imgs)
            splits['val'][clase].extend(val_imgs)
            splits['test'][clase].extend(test_imgs)
            
            print(f"  ✓ {CATEGORIAS[cat_key]['nombre']}/{clase}: "
                  f"{len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    # Mezclar las listas finales para evitar agrupaciones por categoría
    for split in ['train', 'val', 'test']:
        for clase in ['cracked', 'uncracked']:
            random.shuffle(splits[split][clase])
    
    return splits


def copiar_imagenes(splits):
    """
    Copia las imágenes a la estructura de carpetas procesadas.
    
    Args:
        splits (dict): Diccionario con los splits organizados
        
    Returns:
        dict: Estadísticas de archivos copiados
    """
    print(f"\n📋 Paso 3/5: Copiando imágenes a {RUTA_DETECCION}...")
    
    estadisticas = defaultdict(lambda: defaultdict(int))
    
    for split_name in ['train', 'val', 'test']:
        for clase in ['cracked', 'uncracked']:
            # Crear directorio de destino
            destino_dir = RUTA_DETECCION / split_name / clase
            destino_dir.mkdir(parents=True, exist_ok=True)
            
            # Copiar imágenes
            imagenes = splits[split_name][clase]
            for img_path in imagenes:
                destino = destino_dir / img_path.name
                shutil.copy2(img_path, destino)
                estadisticas[split_name][clase] += 1
            
            print(f"  ✓ {split_name}/{clase}: {len(imagenes)} imágenes copiadas")
    
    return estadisticas


def generar_reporte(splits, estadisticas):
    """
    Genera reporte JSON con estadísticas detalladas de la división.
    
    Args:
        splits (dict): Splits generados
        estadisticas (dict): Estadísticas de archivos copiados
        
    Returns:
        dict: Reporte completo
    """
    print("\n📊 Paso 4/5: Generando reporte de estadísticas...")
    
    reporte = {
        'metadata': {
            'fecha_procesamiento': '2025-10-07',
            'semilla_aleatoria': RANDOM_SEED,
            'ratios': {
                'train': TRAIN_RATIO,
                'val': VAL_RATIO,
                'test': TEST_RATIO
            }
        },
        'splits': {},
        'resumen': {}
    }
    
    total_general = 0
    total_cracked = 0
    total_uncracked = 0
    
    # Calcular estadísticas por split
    for split_name in ['train', 'val', 'test']:
        cracked_count = len(splits[split_name]['cracked'])
        uncracked_count = len(splits[split_name]['uncracked'])
        total_count = cracked_count + uncracked_count
        
        reporte['splits'][split_name] = {
            'cracked': cracked_count,
            'uncracked': uncracked_count,
            'total': total_count,
            'porcentaje_cracked': round((cracked_count / total_count * 100), 2) if total_count > 0 else 0,
            'porcentaje_del_dataset': round((total_count / 56092 * 100), 2)
        }
        
        total_general += total_count
        total_cracked += cracked_count
        total_uncracked += uncracked_count
    
    # Resumen general
    reporte['resumen'] = {
        'total_imagenes': total_general,
        'total_cracked': total_cracked,
        'total_uncracked': total_uncracked,
        'porcentaje_cracked_global': round((total_cracked / total_general * 100), 2),
        'verificacion_integridad': total_general == 56092
    }
    
    # Guardar a archivo JSON
    archivo_reporte = Path(config.BASE_DIR) / "reportes" / "splits_info.json"
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=4, ensure_ascii=False)
    
    print(f"  ✓ Reporte guardado en: {archivo_reporte}")
    
    return reporte


def mostrar_resumen(reporte):
    """
    Muestra un resumen visual del proceso completado.
    
    Args:
        reporte (dict): Reporte generado
    """
    print("\n" + "="*70)
    print("✅ DIVISIÓN COMPLETADA EXITOSAMENTE")
    print("="*70)
    
    print(f"\n📊 Resumen de Splits:")
    print(f"  {'Split':<10} {'Total':>10} {'Cracked':>10} {'Uncracked':>12} {'% Cracked':>12}")
    print("  " + "-"*60)
    
    for split_name in ['train', 'val', 'test']:
        datos = reporte['splits'][split_name]
        print(f"  {split_name.upper():<10} {datos['total']:>10} "
              f"{datos['cracked']:>10} {datos['uncracked']:>12} "
              f"{datos['porcentaje_cracked']:>11.2f}%")
    
    print("  " + "-"*60)
    print(f"  {'TOTAL':<10} {reporte['resumen']['total_imagenes']:>10} "
          f"{reporte['resumen']['total_cracked']:>10} "
          f"{reporte['resumen']['total_uncracked']:>12} "
          f"{reporte['resumen']['porcentaje_cracked_global']:>11.2f}%")
    
    print(f"\n✓ Verificación de integridad: {'PASS' if reporte['resumen']['verificacion_integridad'] else 'FAIL'}")
    print(f"✓ Semilla aleatoria: {RANDOM_SEED}")
    print(f"✓ Archivos guardados en: {RUTA_DETECCION}")
    print("\n" + "="*70 + "\n")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que orquesta todo el proceso."""
    
    print("\n" + "="*70)
    print("🚀 PREPROCESAMIENTO SDNET2018 - División Estratificada")
    print("="*70 + "\n")
    
    try:
        # Paso 1: Recolectar imágenes
        imagenes = recolectar_imagenes()
        
        # Paso 2: Dividir estratificadamente
        splits = dividir_estratificado(imagenes)
        
        # Paso 3: Copiar a carpetas procesadas
        estadisticas = copiar_imagenes(splits)
        
        # Paso 4: Generar reporte JSON
        reporte = generar_reporte(splits, estadisticas)
        
        # Paso 5: Mostrar resumen
        mostrar_resumen(reporte)
        
        print("✅ Proceso completado exitosamente.")
        print("📁 Siguiente paso: Verificar los archivos en datos/procesados/deteccion/\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
