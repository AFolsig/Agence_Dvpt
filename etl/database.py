from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///apd.db"

engine = create_engine(DATABASE_URL)

from sqlalchemy import text

def init_db():

    with engine.begin() as conn:

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS processed_batches (
            batch_name TEXT PRIMARY KEY,
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))

def batch_already_loaded(batch_name):

    with engine.begin() as conn:

        result = conn.execute(
            text("""
            SELECT COUNT(*)
            FROM processed_batches
            WHERE batch_name = :name
            """),
            {"name": batch_name}
        )

        return result.scalar() > 0

def mark_batch_loaded(batch_name):

    with engine.begin() as conn:

        conn.execute(
            text("""
            INSERT INTO processed_batches(batch_name)
            VALUES(:name)
            """),
            {"name": batch_name}
        )


def load_batch(df, batch_name):

    df.to_sql(
        "apd_data",
        engine,
        if_exists="append",
        index=False
    )
   
    mark_batch_loaded(batch_name)

    print("Batch chargé")

