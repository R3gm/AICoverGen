import os
import sys

os.system("python src/download_models.py")

args = " ".join(sys.argv[1:])
cmd = f"python src/webui.py {args}"

os.system(cmd)
