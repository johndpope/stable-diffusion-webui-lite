import os
import sys

from install import REPO_PATH, REPO_PATHS

BASE_PATH      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH    = os.path.join(BASE_PATH, "modules")
MODEL_PATH     = os.path.join(BASE_PATH, "models")
EMBEDDING_PATH = os.path.join(BASE_PATH, "embeddings")
SCRIPT_PATH    = os.path.join(BASE_PATH, "scripts")
STATIC_PATH    = os.path.join(BASE_PATH, "static")
RESOURCE_PATH  = os.path.join(BASE_PATH, "resources")


# prepend base path to PATH
sys.path.insert(0, BASE_PATH)

# prepend repo path to PATH
for dp in REPO_PATHS.values:
  sys.path.insert(0, dp)

# prepend module path to PATH
sys.path.insert(0, MODULE_PATH)
