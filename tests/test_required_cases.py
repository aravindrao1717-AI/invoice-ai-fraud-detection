from pathlib import Path
import pandas as pd
from src.fraud_engine import FraudEngine

ROOT = Path(__file__).resolve().parents[1]
HISTORY = pd.read_csv(ROOT / "data" / "synthetic_invoices.csv")
ENGINE = FraudEngine(HISTORY)
BASE = {
    "vendor": "ABC GmbH", "invoice_date": "2026-08-25", "due_date": "2026-09-24",
    "net_amount": 1000.0, "vat_amount": 190.0, "gross_amount": 1190.0, "currency": "EUR",
    "po_number": "PO-5001", "iban": "DE89370400440532013000", "payment_terms": "30 days"
}

def test_normal_invoice_is_verified():
    r = ENGINE.analyze({**BASE, "invoice_number": "INV-NEW-NORMAL-TEST"})
    assert r.risk_level == "VERIFIED"

def test_duplicate_is_high_risk():
    r = ENGINE.analyze({**BASE, "invoice_number": "INV-10001"})
    assert r.risk_level == "HIGH RISK"
    assert any(f["code"] == "DUPLICATE_INVOICE" for f in r.risk_factors)

def test_extreme_amount_is_flagged():
    r = ENGINE.analyze({**BASE, "invoice_number": "INV-EXT-TEST", "net_amount": 21008.4, "vat_amount": 3991.6, "gross_amount": 25000.0})
    assert r.risk_level in {"REQUIRES REVIEW", "HIGH RISK"}
    assert any(f["code"] == "UNUSUAL_AMOUNT" for f in r.risk_factors)

def test_changed_iban_is_high_risk():
    r = ENGINE.analyze({**BASE, "invoice_number": "INV-IBAN-TEST", "iban": "DE12345678901234567890"})
    assert r.risk_level == "HIGH RISK"
    assert any(f["code"] == "IBAN_CHANGE" for f in r.risk_factors)

def test_vat_inconsistency_requires_review():
    r = ENGINE.analyze({**BASE, "invoice_number": "INV-VAT-TEST", "vat_amount": 500.0, "gross_amount": 1500.0})
    assert r.risk_level in {"REQUIRES REVIEW", "HIGH RISK"}
    assert any(f["code"] == "VAT_ANOMALY" for f in r.risk_factors)
