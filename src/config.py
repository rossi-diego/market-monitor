# Python
from pathlib import Path

# Raiz do projeto
DATA_DIR = Path(__file__).parent

# Configs
SRC_DIR = DATA_DIR / "src"

# Data
data_pipeline = SRC_DIR / "data_pipeline.py"
DATA = DATA_DIR.parent / "data" / "commodities_data.csv"

# utils
utils = SRC_DIR / "utils.py"