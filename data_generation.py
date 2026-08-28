from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import random
import numpy as np
import pandas as pd

RNG = random.Random(42)
np.random.seed(42)

VENDORS = [
    ("ABC GmbH", 1000, 0.19, "DE89370400440532013000", "EUR", 30),
    ("TechParts AG", 7800, 0.19, "DE12500105170648489890", "EUR", 14),
    ("Nordic Office Oy", 2200, 0.24, "FI2112345600000785", "EUR", 21),
    ("CloudWorks Ltd", 3500, 0.20, "GB29NWBK60161331926819", "GBP", 30),
    ("DataNova Inc", 6200, 0.00, "US-DEMO-889922", "USD", 30),
    ("Logistik Süd GmbH", 4300, 0.19, "DE44500105175407324931", "EUR", 14),
    ("MediaHaus GmbH", 1800, 0.19, "DE75512108001245126199", "EUR", 30),
    ("Green Supplies BV", 2700, 0.21, "NL91ABNA0417164300", "EUR", 21),
]


def build_dataset(n_normal: int = 180) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 1, 5)
    counters = {v[0]: 1000 for v in VENDORS}
    current_date = {v[0]: start + timedelta(days=RNG.randint(0, 10)) for v in VENDORS}
    for _ in range(n_normal):
        vendor, base, vat_rate, iban, currency, interval = RNG.choice(VENDORS)
        counters[vendor] += 1
        date = current_date[vendor]
        current_date[vendor] = date + timedelta(days=max(3, int(np.random.normal(interval, max(2, interval*0.15)))))
        net = max(50, float(np.random.normal(base, base * 0.11)))
        # For unsupported national VAT profiles, keep historical normal rows plausible but engine demo rates focus on DE.
        effective_vat = vat_rate
        vat = net * effective_vat
        gross = net + vat
        rows.append({
            "vendor": vendor,
            "invoice_number": f"{vendor[:3].upper().replace(' ', '')}-{counters[vendor]}",
            "invoice_date": date.date().isoformat(),
            "due_date": (date + timedelta(days=30)).date().isoformat(),
            "net_amount": round(net, 2),
            "vat_amount": round(vat, 2),
            "gross_amount": round(gross, 2),
            "currency": currency,
            "po_number": f"PO-{RNG.randint(4000, 9999)}",
            "iban": iban,
            "payment_terms": "30 days",
            "synthetic_is_anomaly": False,
            "synthetic_anomaly_types": "normal",
        })

    # Deterministic professor baseline record.
    baseline = {
        "vendor": "ABC GmbH", "invoice_number": "INV-10001", "invoice_date": "2026-07-15", "due_date": "2026-08-14",
        "net_amount": 1000.0, "vat_amount": 190.0, "gross_amount": 1190.0, "currency": "EUR", "po_number": "PO-5001",
        "iban": "DE89370400440532013000", "payment_terms": "30 days", "synthetic_is_anomaly": False, "synthetic_anomaly_types": "normal"
    }
    rows.append(baseline)

    anomalies = [
        {**baseline, "invoice_date": "2026-08-01", "synthetic_is_anomaly": True, "synthetic_anomaly_types": "duplicate"},
        {**baseline, "invoice_number": "INV-EXTREME", "invoice_date": "2026-08-05", "net_amount": 21008.4, "vat_amount": 3991.6, "gross_amount": 25000.0, "synthetic_is_anomaly": True, "synthetic_anomaly_types": "unusual_amount"},
        {**baseline, "invoice_number": "INV-IBAN", "invoice_date": "2026-08-07", "iban": "DE12345678901234567890", "synthetic_is_anomaly": True, "synthetic_anomaly_types": "changed_iban"},
        {**baseline, "invoice_number": "INV-VAT", "invoice_date": "2026-08-09", "net_amount": 1000.0, "vat_amount": 500.0, "gross_amount": 1500.0, "synthetic_is_anomaly": True, "synthetic_anomaly_types": "vat_anomaly"},
        {**baseline, "invoice_number": "INV-NOPO", "invoice_date": "2026-08-11", "net_amount": 25000.0, "vat_amount": 4750.0, "gross_amount": 29750.0, "po_number": "", "synthetic_is_anomaly": True, "synthetic_anomaly_types": "missing_po"},
    ]
    rows.extend(anomalies)

    # Frequency burst.
    for i in range(3):
        rows.append({**baseline, "invoice_number": f"INV-FREQ-{i+1}", "invoice_date": f"2026-08-{20+i:02d}", "synthetic_is_anomaly": True, "synthetic_anomaly_types": "unusual_frequency"})

    df = pd.DataFrame(rows).sort_values("invoice_date").reset_index(drop=True)
    return df


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic_invoices.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
