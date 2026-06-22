import os

from etl.pipeline import transform_batch
from etl.database import (
   load_batch,
   batch_already_loaded,
   init_db
)

BATCH_DIR = "data/batches"


def collect_next_batch():

    init_db()

    files = sorted(
        [
            f for f in os.listdir(BATCH_DIR)
            if f.endswith(".csv")
        ]
    )

    for batch_name in files:

        if not batch_already_loaded(batch_name):

            path = os.path.join(
                BATCH_DIR,
                batch_name
            )

            print(f"Traitement {batch_name}")

            df = transform_batch(path)

            load_batch(df, batch_name)

            return {
                "status": "loaded",
                "batch": batch_name
            }

    return {
        "status": "finished"
    }


if __name__ == "__main__":

    print(collect_next_batch())
