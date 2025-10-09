#!/usr/bin/env python3
"""
Monitor GPU en tiempo real durante entrenamiento
=================================================

Script para monitorear uso de GPU, memoria, throughput y cuellos de botella.
Ejecutar en paralelo mientras entrena el modelo.

Uso:
    python3 scripts/utils/monitor_gpu_wsl.py

Autor: Jesus Naranjo
Fecha: Octubre 2025
"""

import subprocess
import time
import os
from datetime import datetime

def clear_screen():
    """Limpiar terminal"""
    os.system('clear' if os.name == 'posix' else 'cls')

def get_gpu_stats():
    """
    Obtener estadísticas de GPU usando nvidia-smi
    
    Returns:
        dict: Diccionario con métricas de GPU
    """
    try:
        # Comando nvidia-smi con formato CSV
        cmd = [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            # Parsear output
            values = result.stdout.strip().split(', ')
            
            return {
                'gpu_util': float(values[0]),
                'mem_util': float(values[1]),
                'mem_used': int(values[2]),
                'mem_total': int(values[3]),
                'temperature': int(values[4]),
                'power_draw': float(values[5]),
                'power_limit': float(values[6])
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error obteniendo stats GPU: {e}")
        return None


def get_process_info():
    """
    Obtener información de procesos usando GPU
    
    Returns:
        list: Lista de procesos con uso de GPU
    """
    try:
        cmd = [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 and result.stdout.strip():
            processes = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    processes.append({
                        'pid': int(parts[0]),
                        'name': parts[1],
                        'mem_mb': int(parts[2])
                    })
            return processes
        return []
        
    except Exception as e:
        print(f"Error obteniendo procesos: {e}")
        return []


def format_bar(value, max_value, width=40, char='█'):
    """
    Crear barra de progreso visual
    
    Args:
        value (float): Valor actual
        max_value (float): Valor máximo
        width (int): Ancho de la barra
        char (str): Carácter para la barra
        
    Returns:
        str: Barra formateada
    """
    filled = int((value / max_value) * width)
    bar = char * filled + '░' * (width - filled)
    return bar


def get_color(value, thresholds={'green': 30, 'yellow': 70, 'red': 90}):
    """
    Obtener código ANSI de color según valor
    
    Args:
        value (float): Valor a colorear
        thresholds (dict): Umbrales de color
        
    Returns:
        str: Código de color ANSI
    """
    if value < thresholds['green']:
        return '\033[92m'  # Verde
    elif value < thresholds['yellow']:
        return '\033[93m'  # Amarillo
    else:
        return '\033[91m'  # Rojo


def display_stats(stats, processes, iteration):
    """
    Mostrar estadísticas formateadas en terminal
    
    Args:
        stats (dict): Estadísticas de GPU
        processes (list): Lista de procesos
        iteration (int): Número de iteración
    """
    clear_screen()
    
    reset = '\033[0m'
    bold = '\033[1m'
    
    print("=" * 80)
    print(f"{bold}🎮 MONITOR GPU - NVIDIA RTX 2050{reset}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Iteración: {iteration}")
    print("=" * 80)
    
    if stats is None:
        print("❌ No se pudo obtener información de GPU")
        print("   Verifica que nvidia-smi esté disponible en WSL2")
        return
    
    # GPU Utilization
    gpu_color = get_color(stats['gpu_util'])
    gpu_bar = format_bar(stats['gpu_util'], 100)
    print(f"\n📊 {bold}GPU Utilization:{reset}")
    print(f"   {gpu_color}{gpu_bar}{reset} {stats['gpu_util']:.1f}%")
    
    # Diagnóstico de GPU utilization
    if stats['gpu_util'] < 30:
        print(f"   ⚠️  {bold}BAJA UTILIZACIÓN{reset} - Posible cuello de botella en CPU/I/O")
    elif stats['gpu_util'] < 70:
        print(f"   ⚡ Utilización moderada - Aún hay margen de mejora")
    else:
        print(f"   ✅ {bold}EXCELENTE{reset} - GPU bien aprovechada")
    
    # Memory Utilization
    mem_pct = (stats['mem_used'] / stats['mem_total']) * 100
    mem_color = get_color(mem_pct)
    mem_bar = format_bar(stats['mem_used'], stats['mem_total'])
    print(f"\n💾 {bold}VRAM Usage:{reset}")
    print(f"   {mem_color}{mem_bar}{reset} {stats['mem_used']} MB / {stats['mem_total']} MB ({mem_pct:.1f}%)")
    
    if mem_pct > 90:
        print(f"   ⚠️  {bold}VRAM CASI LLENA{reset} - Riesgo OOM, considera reducir batch_size")
    elif mem_pct > 75:
        print(f"   ⚡ VRAM en uso alto - Monitorear de cerca")
    else:
        print(f"   ✅ VRAM con margen suficiente")
    
    # Temperature
    temp_color = get_color(stats['temperature'], {'green': 60, 'yellow': 75, 'red': 85})
    temp_bar = format_bar(stats['temperature'], 100)
    print(f"\n🌡️  {bold}Temperature:{reset}")
    print(f"   {temp_color}{temp_bar}{reset} {stats['temperature']}°C")
    
    if stats['temperature'] > 85:
        print(f"   ⚠️  {bold}TEMPERATURA ALTA{reset} - Verifica ventilación")
    elif stats['temperature'] > 75:
        print(f"   ⚡ Temperatura elevada - Normal bajo carga")
    else:
        print(f"   ✅ Temperatura saludable")
    
    # Power Draw
    power_pct = (stats['power_draw'] / stats['power_limit']) * 100
    power_color = get_color(power_pct)
    power_bar = format_bar(stats['power_draw'], stats['power_limit'])
    print(f"\n⚡ {bold}Power Draw:{reset}")
    print(f"   {power_color}{power_bar}{reset} {stats['power_draw']:.1f}W / {stats['power_limit']:.1f}W ({power_pct:.1f}%)")
    
    # Processes
    print(f"\n🔧 {bold}Procesos usando GPU:{reset}")
    if processes:
        for proc in processes:
            print(f"   PID {proc['pid']}: {proc['name']} - {proc['mem_mb']} MB")
    else:
        print(f"   ⚠️  No hay procesos detectados usando GPU")
    
    # Recomendaciones
    print(f"\n💡 {bold}DIAGNÓSTICO:{reset}")
    
    if stats['gpu_util'] < 30 and stats['mem_used'] < stats['mem_total'] * 0.5:
        print(f"   🔴 GPU INFRAUTILIZADA - Posibles causas:")
        print(f"      - Cuello de botella en I/O de disco (datos no llegan rápido)")
        print(f"      - Batch size muy pequeño")
        print(f"      - Pipeline CPU-GPU no optimizado")
        print(f"   📝 SOLUCIONES:")
        print(f"      1. Aumentar BATCH_SIZE (actual: 16 → probar 24-32)")
        print(f"      2. Aumentar PREFETCH_BUFFER en script (actual: 3 → probar 5-8)")
        print(f"      3. Copiar datos a /tmp en WSL (evita OneDrive)")
    
    elif stats['gpu_util'] > 85 and mem_pct < 80:
        print(f"   🟢 GPU BIEN OPTIMIZADA")
        print(f"   ✅ Utilización excelente, VRAM con margen")
        print(f"   💡 Podrías aumentar BATCH_SIZE para aprovechar más VRAM")
    
    elif mem_pct > 90:
        print(f"   🟡 VRAM CRÍTICA")
        print(f"   ⚠️  Reducir BATCH_SIZE para evitar OOM errors")
    
    else:
        print(f"   🟡 RENDIMIENTO ACEPTABLE")
        print(f"   ⚡ GPU trabajando, pero con margen de optimización")
    
    print("=" * 80)
    print(f"Presiona Ctrl+C para detener monitor | Actualización cada 2 segundos")
    print("=" * 80)


def main():
    """
    Loop principal de monitoreo
    """
    print("Iniciando monitor GPU...")
    print("Presiona Ctrl+C para detener\n")
    
    iteration = 0
    
    try:
        while True:
            stats = get_gpu_stats()
            processes = get_process_info()
            
            display_stats(stats, processes, iteration)
            
            iteration += 1
            time.sleep(2)  # Actualizar cada 2 segundos
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitor detenido por usuario")
        print("=" * 80)


if __name__ == "__main__":
    main()
