from pathlib import Path
import sqlite3
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    invoice_date TEXT NOT NULL,
    due_date TEXT,
    net_amount REAL NOT NULL,
    vat_amount REAL NOT NULL,
    gross_amount REAL NOT NULL,
    currency TEXT NOT NULL,
    po_number TEXT,
    iban TEXT,
    payment_terms TEXT,
    synthetic_is_anomaly INTEGER DEFAULT 0,
    synthetic_anomaly_types TEXT DEFAULT 'normal'
);
"""

def initialize_database(db_path: Path, seed_csv: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute(SCHEMA)
        count = con.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
        if count == 0:
            df = pd.read_csv(seed_csv)
            cols = [
                "vendor","invoice_number","invoice_date","due_date","net_amount","vat_amount","gross_amount","currency",
                "po_number","iban","payment_terms","synthetic_is_anomaly","synthetic_anomaly_types"
            ]
            df[cols].to_sql("invoices", con, if_exists="append", index=False)

def load_invoices(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM invoices ORDER BY invoice_date, id", con)
