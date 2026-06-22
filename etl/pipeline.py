import pandas as pd


def transform_batch(batch_path):
   """
   Lecture + nettoyage
   """

   df = pd.read_csv(batch_path)

   df.columns = [
       c.strip().replace(" ", "_")
       for c in df.columns
   ]

   return df
