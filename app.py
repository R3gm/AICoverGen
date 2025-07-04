import os

cmd = """

pip install onnxruntime-gpu[cuda,cudnn]==1.22.0
find / -name 'libcudnn.so*' 2>/dev/null

python src/download_models.py
python src/webui.py
"""

os.system(cmd)
