# Python
from pathlib import Path

# Raiz do projeto
BASE_DIR = Path(__file__).parent

# Configs
SRC_DIR = BASE_DIR / "src"

# Data
data_pipeline = SRC_DIR / "data_pipeline.py"
BASE = BASE_DIR.parent / "data" / "commodities_data.csv"

# utils
utils = SRC_DIR / "utils.py"