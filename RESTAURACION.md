# 🚀 Guía Rápida de Restauración de Versiones

## Restaurar a Versión Estable (v1.0-production)

### Opción 1: Restauración Rápida (Recomendada)

```bash
# Volver al estado estable
git checkout backup/v1.0-production

# Crear nueva rama de trabajo si necesitas hacer cambios
git checkout -b hotfix/nombre-descriptivo
```

### Opción 2: Con Tag

```bash
git checkout v1.0-production

# Para trabajar sobre este estado
git checkout -b hotfix/nombre-descriptivo v1.0-production
```

### Opción 3: Reset Completo (⚠️ Cuidado)

```bash
# Solo si quieres descartar TODO el trabajo posterior
git reset --hard v1.0-production

# Forzar push (⚠️ destructivo)
git push -f origin main
```

---

## Verificar Versión Actual

```bash
# Ver en qué rama estás
git branch

# Ver último commit
git log --oneline -1

# Ver todos los tags disponibles
git tag

# Ver todas las ramas de backup
git branch -a | grep backup
```

---

## Comparar Versiones

```bash
# Ver diferencias entre v1.0 y tu rama actual
git diff v1.0-production

# Ver solo nombres de archivos cambiados
git diff v1.0-production --name-only

# Ver estadísticas de cambios
git diff v1.0-production --stat
```

---

## Comandos de Emergencia

### Si el sistema está roto en main:

```bash
# 1. Crear rama de rescate del estado actual
git branch rescue/estado-roto

# 2. Volver a v1.0 estable
git checkout backup/v1.0-production

# 3. Forzar main a v1.0
git checkout main
git reset --hard backup/v1.0-production
git push -f origin main

# 4. Streamlit Cloud se re-desplegará automáticamente con v1.0
```

### Si necesitas recuperar un archivo específico de v1.0:

```bash
# Recuperar solo app.py de v1.0
git checkout v1.0-production -- app_web/app.py

# Recuperar solo requirements.txt
git checkout v1.0-production -- app_web/requirements.txt
```

---

## Crear Nueva Versión

```bash
# 1. Asegurarte de estar en main actualizado
git checkout main
git pull

# 2. Crear tag de nueva versión
git tag -a v1.1-ensemble -m "Descripción de la versión"

# 3. Crear rama de backup
git branch backup/v1.1-ensemble

# 4. Subir todo a GitHub
git push origin v1.1-ensemble
git push origin backup/v1.1-ensemble
```

---

## Workflows Comunes

### Desarrollo de nueva feature:

```bash
# 1. Crear rama desde main
git checkout -b feature/nombre-feature

# 2. Hacer cambios y commits
git add .
git commit -m "feat: descripción"

# 3. Testing local
streamlit run app_web/app.py

# 4. Si funciona, mergear a main
git checkout main
git merge feature/nombre-feature

# 5. Crear tag y backup
git tag -a v1.x-nombre
git branch backup/v1.x-nombre
git push --all
git push --tags
```

### Hotfix urgente en producción:

```bash
# 1. Crear rama desde v1.0 estable
git checkout -b hotfix/descripcion v1.0-production

# 2. Fix rápido
# ... hacer cambios ...
git commit -m "hotfix: descripción"

# 3. Mergear a main
git checkout main
git merge hotfix/descripcion

# 4. Tag de parche
git tag -a v1.0.1 -m "Hotfix: descripción"
git push --all --tags
```

---

## Estados del Proyecto

```
🟢 v1.0-production     → ESTABLE, en producción
🟡 v1.1-ensemble       → TESTING, desarrollo
🔴 v2.0-retrained      → PLANIFICADA, futuro
```

---

## ⚠️ IMPORTANTE

- **SIEMPRE** hacer backup antes de cambios grandes
- **SIEMPRE** probar localmente antes de push a main
- **SIEMPRE** crear tag + rama backup para versiones estables
- **NUNCA** hacer `git push -f` sin backup previo
- **NUNCA** eliminar tags sin consultar

---

**Última actualización:** 12 Diciembre 2025
