# Comandos para Sincronizar Archivos al Servidor

Este archivo contiene los comandos necesarios para copiar los archivos actualizados a tu servidor remoto.

## Opción 1: Usando SCP (recomendado)

```bash
# Desde tu máquina local (Windows PowerShell o Linux/Mac terminal)

# Define las variables del servidor
SERVER_USER="tu_usuario"
SERVER_HOST="tu_servidor.com"
SERVER_PATH="/home/SECURE_DATA/PAU"

# Copiar test_video.py
scp tests/test_video.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/tests/

# Copiar test_all_videos.py
scp tests/test_all_videos.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/tests/

# Copiar pre_encode_videos.py
scp tests/pre_encode_videos.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/tests/

# Copiar script de verificación
scp tests/check_preencoded_support.py ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/tests/

# Verificar en el servidor
ssh ${SERVER_USER}@${SERVER_HOST} "cd ${SERVER_PATH} && python tests/check_preencoded_support.py"
```

## Opción 2: Usando SFTP

```bash
# Conectar al servidor
sftp tu_usuario@tu_servidor.com

# Navegar al directorio
cd /home/SECURE_DATA/PAU/tests

# Subir archivos
put tests/test_video.py
put tests/test_all_videos.py
put tests/pre_encode_videos.py
put tests/check_preencoded_support.py

# Salir
exit
```

## Opción 3: Usando Git (si tienes repositorio)

```bash
# En tu máquina local
git add tests/test_video.py tests/test_all_videos.py tests/pre_encode_videos.py
git commit -m "Add pre-encoded video support"
git push origin main

# En el servidor
ssh tu_usuario@tu_servidor.com
cd /home/SECURE_DATA/PAU
git pull origin main

# Verificar
python tests/check_preencoded_support.py
```

## Opción 4: Copiar Manualmente (Windows)

Si usas WinSCP, FileZilla u otro cliente FTP/SFTP:

1. Conecta a tu servidor
2. Navega a `/home/SECURE_DATA/PAU/tests/`
3. Sube estos archivos:
   - `test_video.py`
   - `test_all_videos.py`
   - `pre_encode_videos.py`
   - `check_preencoded_support.py`

## Verificación

Después de copiar los archivos, conéctate al servidor y ejecuta:

```bash
ssh tu_usuario@tu_servidor.com
cd /home/SECURE_DATA/PAU
python tests/check_preencoded_support.py
```

Deberías ver:
```
✅ SUCCESS: test_video.py has the --use-preencoded argument
✅ SUCCESS: load_preencoded_video function is present
✅ SUCCESS: Path import is present
✅ SUCCESS: test_all_videos.py has the --use-preencoded argument and passes it correctly
✅ SUCCESS: pre_encode_videos.py exists
✅ ALL CHECKS PASSED - Your files are up to date!
```

## Si los Checks Fallan

Si ves errores después de copiar:

1. **Verifica permisos de archivos:**
   ```bash
   chmod +x tests/*.py
   ```

2. **Compara checksums** (para asegurar que la copia fue exitosa):
   ```bash
   # En local (Windows PowerShell)
   Get-FileHash tests/test_video.py -Algorithm MD5
   
   # En servidor (Linux)
   md5sum tests/test_video.py
   ```

3. **Verifica que no hay problemas de encoding:**
   ```bash
   file tests/test_video.py
   # Debería mostrar: UTF-8 Unicode text
   ```

## Ejemplo Completo (PowerShell en Windows)

```powershell
# Configurar variables
$SERVER_USER = "tu_usuario"
$SERVER_HOST = "tu_servidor.com"
$SERVER_PATH = "/home/SECURE_DATA/PAU"

# Función para copiar y verificar
function Sync-FileToServer {
    param($LocalFile, $RemotePath)
    
    Write-Host "Copiando $LocalFile..." -ForegroundColor Yellow
    scp $LocalFile "${SERVER_USER}@${SERVER_HOST}:${RemotePath}"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $LocalFile copiado exitosamente" -ForegroundColor Green
    } else {
        Write-Host "❌ Error copiando $LocalFile" -ForegroundColor Red
    }
}

# Copiar archivos
Sync-FileToServer "tests/test_video.py" "${SERVER_PATH}/tests/"
Sync-FileToServer "tests/test_all_videos.py" "${SERVER_PATH}/tests/"
Sync-FileToServer "tests/pre_encode_videos.py" "${SERVER_PATH}/tests/"
Sync-FileToServer "tests/check_preencoded_support.py" "${SERVER_PATH}/tests/"

# Verificar en el servidor
Write-Host "`nVerificando archivos en el servidor..." -ForegroundColor Yellow
ssh "${SERVER_USER}@${SERVER_HOST}" "cd ${SERVER_PATH} && python tests/check_preencoded_support.py"
```

## Notas Importantes

- Asegúrate de tener permisos SSH/SCP configurados
- Si usas autenticación por clave, ten tu clave SSH lista
- Los paths pueden variar según tu configuración del servidor
- Mantén backups de tus archivos originales por si acaso
