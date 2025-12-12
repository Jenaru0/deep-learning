# 🚀 COMANDOS GIT PARA DEPLOY

## PASO 1: Verificar qué se va a subir

```bash
# Ver tamaño del proyecto (sin datasets/venv)
git status
git ls-files | xargs -I {} du -h {} | sort -hr | head -20
```

## PASO 2: Inicializar Git (si no está inicializado)

```bash
cd "/mnt/c/Users/jonna/OneDrive/Escritorio/DEEP LEARNING/investigacion_fisuras"

# Inicializar repo
git init

# Agregar origin (reemplaza TU_USUARIO)
git remote add origin https://github.com/Jenaru0/investigacion_fisuras.git
```

## PASO 3: Agregar archivos (respetando .gitignore)

```bash
# Ver qué se va a agregar
git add --dry-run .

# Si todo está bien, agregar realmente
git add .

# Ver status
git status
```

## PASO 4: Commit y Push

```bash
# Crear commit
git commit -m "Deploy: Sistema de detección de fisuras con Streamlit"

# Crear rama main
git branch -M main

# Push a GitHub
git push -u origin main
```

## PASO 5: Verificar tamaño en GitHub

```bash
# Ver tamaño del repo
git count-objects -vH
```

---

## ⚠️ SI GIT RECHAZA POR TAMAÑO:

Si algún archivo supera 100 MB:

```bash
# Ver archivos grandes
find . -type f -size +10M -exec ls -lh {} \;

# Usar Git LFS para archivos grandes
git lfs install
git lfs track "*.keras"
git add .gitattributes
git commit -m "Agregar Git LFS"
git push
```

---

## 🔧 COMANDOS DE EMERGENCIA:

```bash
# Deshacer último commit (si te equivocaste)
git reset --soft HEAD~1

# Limpiar archivos no trackeados
git clean -fd

# Ver archivos ignorados
git status --ignored
```

---

## ✅ CHECKLIST FINAL:

- [ ] `.gitignore` actualizado
- [ ] `requirements_streamlit.txt` creado
- [ ] `.streamlit/config.toml` creado
- [ ] `download_models.py` con IDs de Drive
- [ ] `app.py` modificado para cloud
- [ ] Commit realizado
- [ ] Push a GitHub exitoso
- [ ] Repo público en GitHub
