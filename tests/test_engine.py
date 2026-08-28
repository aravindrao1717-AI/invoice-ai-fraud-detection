from pathlib import Path
import pandas as pd
from src.fraud_engine import FraudEngine

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_has_at_least_100_records():
    df = pd.read_csv(ROOT / "data" / "synthetic_invoices.csv")
    assert len(df) >= 100

def test_missing_high_value_po_is_flagged():
    df = pd.read_csv(ROOT / "data" / "synthetic_invoices.csv")
    engine = FraudEngine(df)
    payload = {"vendor":"ABC GmbH","invoice_number":"NOPO-X","invoice_date":"2026-08-25","due_date":"2026-09-24","net_amount":25000,"vat_amount":4750,"gross_amount":29750,"currency":"EUR","po_number":"","iban":"DE89370400440532013000","payment_terms":"30 days"}
    result = engine.analyze(payload)
    assert any(f["code"] == "MISSING_PO" for f in result.risk_factors)
