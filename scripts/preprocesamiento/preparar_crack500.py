"""
Script de Preprocesamiento: Preparación de CRACK500 para Segmentación
======================================================================
Lee los splits predefinidos de CRACK500 (train/val/test) y copia imágenes
y máscaras a la estructura procesada, validando correspondencia 1:1.

Autor: Sistema de Detección de Fisuras
Fecha: 8 de octubre, 2025
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

# Agregar la ruta del proyecto al sys.path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RUTA_CRACK500 = Path(config.RUTA_CRACK500)
RUTA_SEGMENTACION = Path(config.RUTA_SEGMENTACION)
METADATA_DIR = RUTA_CRACK500 / "metadata"

# Directorios de origen
IMAGES_DIR = RUTA_CRACK500 / "images"
# NOTA: Las máscaras también están en el directorio "images/" (formato .png)
MASKS_DIR = IMAGES_DIR  # Las máscaras están mezcladas con las imágenes

# Archivos de splits predefinidos
SPLITS_FILES = {
    'train': METADATA_DIR / "train.txt",
    'val': METADATA_DIR / "val.txt",
    'test': METADATA_DIR / "test.txt"
}

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def leer_split(archivo_split):
    """
    Lee un archivo de split y retorna pares de rutas (imagen, máscara).
    
    Formato esperado en archivo: "ruta/imagen.jpg ruta/mascara.png"
    
    Args:
        archivo_split (Path): Ruta al archivo .txt con pares de archivos
        
    Returns:
        list: Lista de tuplas (nombre_imagen_sin_ext, ruta_img, ruta_mask)
    """
    if not archivo_split.exists():
        raise FileNotFoundError(f"Archivo de split no encontrado: {archivo_split}")
    
    pares = []
    with open(archivo_split, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Formato: "traincrop/20160222_081011_1281_721.jpg traincrop/20160222_081011_1281_721.png"
            partes = line.split()
            if len(partes) == 2:
                img_path, mask_path = partes
                # Extraer solo el nombre base sin extensión
                nombre = Path(img_path).stem
                pares.append((nombre, img_path, mask_path))
    
    return pares


def validar_existencia_archivos(pares, split_name):
    """
    Valida que cada imagen tenga su máscara correspondiente.
    
    Args:
        pares (list): Lista de tuplas (nombre, ruta_img, ruta_mask)
        split_name (str): Nombre del split (train/val/test)
        
    Returns:
        dict: Diccionario con archivos válidos, faltantes y huérfanos
    """
    print(f"\n🔍 Validando archivos del split '{split_name}'...")
    
    validos = []
    imagenes_faltantes = []
    mascaras_faltantes = []
    
    for nombre, img_rel_path, mask_rel_path in pares:
        # Construir rutas absolutas
        img_path = IMAGES_DIR / Path(img_rel_path).name  # Solo nombre de archivo
        mask_path = MASKS_DIR / Path(mask_rel_path).name
        
        img_existe = img_path.exists()
        mask_existe = mask_path.exists()
        
        if img_existe and mask_existe:
            validos.append((nombre, img_path, mask_path))
        elif not img_existe:
            imagenes_faltantes.append(nombre)
        elif not mask_existe:
            mascaras_faltantes.append(nombre)
    
    # Reporte de validación
    print(f"  ✓ Pares válidos (imagen + máscara): {len(validos)}")
    
    if imagenes_faltantes:
        print(f"  ⚠️  Imágenes faltantes: {len(imagenes_faltantes)}")
        if len(imagenes_faltantes) <= 5:
            print(f"     {imagenes_faltantes}")
        else:
            print(f"     Primeras 5: {imagenes_faltantes[:5]}")
    
    if mascaras_faltantes:
        print(f"  ⚠️  Máscaras faltantes: {len(mascaras_faltantes)}")
        if len(mascaras_faltantes) <= 5:
            print(f"     {mascaras_faltantes}")
        else:
            print(f"     Primeras 5: {mascaras_faltantes[:5]}")
    
    return {
        'validos': validos,
        'imagenes_faltantes': imagenes_faltantes,
        'mascaras_faltantes': mascaras_faltantes
    }


def copiar_archivos(split_name, pares_validos):
    """
    Copia imágenes y máscaras al directorio de segmentación procesada.
    
    Args:
        split_name (str): Nombre del split (train/val/test)
        pares_validos (list): Lista de tuplas (nombre, img_path, mask_path)
        
    Returns:
        int: Cantidad de pares copiados
    """
    print(f"\n📋 Copiando archivos del split '{split_name}'...")
    
    # Crear directorios de destino
    destino_images = RUTA_SEGMENTACION / split_name / "images"
    destino_masks = RUTA_SEGMENTACION / split_name / "masks"
    
    destino_images.mkdir(parents=True, exist_ok=True)
    destino_masks.mkdir(parents=True, exist_ok=True)
    
    # Copiar cada par imagen-máscara
    copiados = 0
    for nombre, src_img, src_mask in pares_validos:
        # Copiar imagen
        dst_img = destino_images / src_img.name
        shutil.copy2(src_img, dst_img)
        
        # Copiar máscara
        dst_mask = destino_masks / src_mask.name
        shutil.copy2(src_mask, dst_mask)
        
        copiados += 1
    
    print(f"  ✓ {copiados} pares imagen-máscara copiados")
    print(f"    - Imágenes: {destino_images}")
    print(f"    - Máscaras: {destino_masks}")
    
    return copiados


def generar_reporte(estadisticas_splits):
    """
    Genera reporte JSON con estadísticas completas de CRACK500.
    
    Args:
        estadisticas_splits (dict): Diccionario con estadísticas por split
        
    Returns:
        dict: Reporte completo
    """
    print("\n📊 Generando reporte de estadísticas...")
    
    reporte = {
        'metadata': {
            'dataset': 'CRACK500',
            'tarea': 'Segmentación de fisuras',
            'fecha_procesamiento': '2025-10-08',
            'splits_predefinidos': True,
            'fuente_splits': 'metadata/train.txt, val.txt, test.txt'
        },
        'splits': {},
        'validacion': {},
        'resumen': {}
    }
    
    total_validos = 0
    total_img_faltantes = 0
    total_mask_faltantes = 0
    
    # Estadísticas por split
    for split_name, stats in estadisticas_splits.items():
        validos = len(stats['validos'])
        img_faltantes = len(stats['imagenes_faltantes'])
        mask_faltantes = len(stats['mascaras_faltantes'])
        
        reporte['splits'][split_name] = {
            'pares_validos': validos,
            'imagenes_faltantes': img_faltantes,
            'mascaras_faltantes': mask_faltantes,
            'total_esperado': validos + img_faltantes + mask_faltantes
        }
        
        reporte['validacion'][split_name] = {
            'integridad': img_faltantes == 0 and mask_faltantes == 0,
            'archivos_problematicos': stats['imagenes_faltantes'] + stats['mascaras_faltantes']
        }
        
        total_validos += validos
        total_img_faltantes += img_faltantes
        total_mask_faltantes += mask_faltantes
    
    # Resumen general
    reporte['resumen'] = {
        'total_pares_validos': total_validos,
        'total_imagenes_faltantes': total_img_faltantes,
        'total_mascaras_faltantes': total_mask_faltantes,
        'integridad_completa': total_img_faltantes == 0 and total_mask_faltantes == 0,
        'comparacion_inventario': {
            'esperado_total': config.CRACK500_INVENTORY['total'],
            'procesado_total': total_validos,
            'match': total_validos == config.CRACK500_INVENTORY['total']
        }
    }
    
    # Guardar a archivo JSON
    archivo_reporte = Path(config.BASE_DIR) / "reportes" / "crack500_info.json"
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
    print("✅ PREPARACIÓN DE CRACK500 COMPLETADA")
    print("="*70)
    
    print(f"\n📊 Resumen de Splits:")
    print(f"  {'Split':<10} {'Válidos':>10} {'Img Faltantes':>15} {'Mask Faltantes':>16} {'Integridad':>12}")
    print("  " + "-"*70)
    
    for split_name in ['train', 'val', 'test']:
        datos = reporte['splits'][split_name]
        validacion = reporte['validacion'][split_name]
        integridad = "✓ OK" if validacion['integridad'] else "✗ FAIL"
        
        print(f"  {split_name.upper():<10} {datos['pares_validos']:>10} "
              f"{datos['imagenes_faltantes']:>15} "
              f"{datos['mascaras_faltantes']:>16} "
              f"{integridad:>12}")
    
    print("  " + "-"*70)
    resumen = reporte['resumen']
    print(f"  {'TOTAL':<10} {resumen['total_pares_validos']:>10} "
          f"{resumen['total_imagenes_faltantes']:>15} "
          f"{resumen['total_mascaras_faltantes']:>16}")
    
    print(f"\n✓ Integridad completa: {'SÍ' if resumen['integridad_completa'] else 'NO'}")
    print(f"✓ Comparación con inventario: {resumen['total_pares_validos']} de {resumen['comparacion_inventario']['esperado_total']} esperados")
    print(f"✓ Archivos guardados en: {RUTA_SEGMENTACION}")
    print("\n" + "="*70 + "\n")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que orquesta todo el proceso."""
    
    print("\n" + "="*70)
    print("🚀 PREPROCESAMIENTO CRACK500 - Preparación para Segmentación")
    print("="*70 + "\n")
    
    try:
        estadisticas_splits = {}
        
        # Procesar cada split
        for split_name, split_file in SPLITS_FILES.items():
            print(f"\n{'='*70}")
            print(f"Procesando split: {split_name.upper()}")
            print(f"{'='*70}")
            
            # Paso 1: Leer pares de archivos del split
            print(f"\n📂 Paso 1/3: Leyendo archivo {split_file.name}...")
            pares = leer_split(split_file)
            print(f"  ✓ {len(pares)} pares listados en {split_file.name}")
            
            # Paso 2: Validar existencia de imágenes y máscaras
            validacion = validar_existencia_archivos(pares, split_name)
            estadisticas_splits[split_name] = validacion
            
            # Paso 3: Copiar archivos válidos
            if validacion['validos']:
                copiar_archivos(split_name, validacion['validos'])
            else:
                print(f"  ⚠️  No hay archivos válidos para copiar en {split_name}")
        
        # Generar reporte final
        print(f"\n{'='*70}")
        print("Generando Reporte Final")
        print(f"{'='*70}")
        
        reporte = generar_reporte(estadisticas_splits)
        mostrar_resumen(reporte)
        
        # Verificar si hubo problemas
        if not reporte['resumen']['integridad_completa']:
            print("⚠️  ADVERTENCIA: Se encontraron archivos faltantes.")
            print("   Revisa reportes/crack500_info.json para más detalles.\n")
        else:
            print("✅ Proceso completado exitosamente sin errores.\n")
        
        print("📁 Siguiente paso: Entrenar modelo de segmentación U-Net\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
