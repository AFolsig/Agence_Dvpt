import pandas as pd
from pathlib import Path

RAW_PATH = "data/raw/aide-publique-au-developpement.csv"
OUTPUT_DIR = Path("data/batches")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def split_csv(chunk_size=10000):
   for i, chunk in enumerate(pd.read_csv(RAW_PATH, sep=";", chunksize=chunk_size, low_memory=False)):
       output_path = OUTPUT_DIR / f"batch_{i+1}.csv"
       chunk.to_csv(output_path, index=False)
       print(f"Batch créé : {output_path}")

if __name__ == "__main__":
   split_csv()
