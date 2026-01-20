import os
import pickle
import numpy as np
from PIL import Image
import subprocess
import shutil

# ==============================
# CONFIGURAÇÃO
# ==============================
PKL_PATH = "mnist_test_15percent.pkl"
OUTPUT_DIR = "mnist_images_test"
RAR_NAME = "mnist_images_test.rar"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 1) CARREGAR PKL
# ==============================
print("📂 Carregando arquivo PKL...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

print("✅ PKL carregado")
print("Chaves:", list(data.keys()))

# ==============================
# 2) DETECTAR IMAGENS E LABELS
# ==============================
X = None
y = None

for value in data.values():
    arr = np.array(value)

    # imagens
    if arr.ndim in [2, 3, 4] and arr.shape[0] > 100:
        if arr.ndim == 2 and arr.shape[1] == 784:
            X = arr
        elif arr.ndim >= 3 and arr.shape[1] == 28 and arr.shape[2] == 28:
            X = arr

    # labels
    if arr.ndim == 1 and arr.dtype.kind in "iu" and arr.max() <= 9:
        y = arr

if X is None or y is None:
    raise RuntimeError("❌ Não foi possível identificar imagens e labels")

print(f"Imagens: {X.shape}")
print(f"Labels: {y.shape}")

# ==============================
# 3) AJUSTAR FORMATO
# ==============================
if X.ndim == 2:
    X = X.reshape(-1, 28, 28)

if X.ndim == 4 and X.shape[-1] == 1:
    X = X.squeeze(-1)

# ==============================
# 4) SALVAR IMAGENS EM PASTAS
# ==============================
print("💾 Salvando imagens...")

class_counts = {i: 0 for i in range(10)}

for i, (img, label) in enumerate(zip(X, y)):
    class_dir = os.path.join(OUTPUT_DIR, str(label))
    os.makedirs(class_dir, exist_ok=True)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    Image.fromarray(img).save(
        os.path.join(class_dir, f"img_{i:05d}.png")
    )

    class_counts[label] += 1

# ==============================
# 5) CRIAR ARQUIVO RAR
# ==============================
print("📦 Compactando em RAR...")

RAR_PATH = os.path.abspath(RAR_NAME)
FOLDER_PATH = os.path.abspath(OUTPUT_DIR)

rar_exe = r"C:\Program Files\WinRAR\rar.exe"

if not os.path.exists(rar_exe):
    raise RuntimeError("❌ WinRAR não encontrado. Instale o WinRAR.")

subprocess.run(
    [rar_exe, "a", "-r", RAR_PATH, FOLDER_PATH],
    check=True
)

print(f"✅ Arquivo RAR criado: {RAR_PATH}")

# ==============================
# 6) (OPCIONAL) REMOVER PASTA
# ==============================
# Descomente se quiser manter apenas o RAR
# shutil.rmtree(OUTPUT_DIR)

# ==============================
# 7) RESUMO
# ==============================
print("\nResumo por classe:")
for digit in range(10):
    print(f"  {digit}: {class_counts[digit]} imagens")
