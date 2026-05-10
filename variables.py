import os

# Root directory of the data
ROOT_DIR = "./deltas"

# Subdirectory names for each delta type
DELTA_DIRS = ["aoa", "freq", "phon", "conc"]

# Word-pair file name (adjust if different)
WORDPAIR_FILENAME = "word_pairs.txt"

# Output directory for figures and tables
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTORS = ["delta_freq", "delta_conc", "delta_phon"]
OUTCOME = "delta_aoa"

# The two string columns identifying a word pair inside each parquet file
WORD_I_COL = "word_i"
WORD_J_COL = "word_j"

# Numeric columns holding the delta value inside each parquet file
DELTA_COL = "delta"