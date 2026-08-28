# Invoice AI – Intelligent Invoice Fraud Detection

A compact Streamlit web application for explainable invoice fraud/risk assessment. It supports manual invoice entry, CSV batch upload, dashboard metrics, synthetic invoice history, deterministic finance controls, and an Isolation Forest anomaly signal.

> **Important:** The displayed 0–100 value is a **risk score**, not a calibrated probability of fraud.

## Live Demo

Deploy this repository to Streamlit Community Cloud and place the public URL here before submission:

`https://invoice-ai-fraud-detection.streamlit.app/`

## 1. Project Architecture

```text
Streamlit UI (app.py)
        |
        +--> Manual invoice analysis
        +--> CSV batch analysis
        +--> Dashboard / invoice history
        |
        v
Hybrid Fraud Engine (src/fraud_engine.py)
        |
        +--> Deterministic controls
        |      - duplicate invoice
        |      - unusual vendor-relative amount
        |      - VAT anomaly / total mismatch
        |      - missing purchase order
        |      - changed IBAN
        |      - unusual invoice frequency
        |      - new vendor / limited history
        |
        +--> Isolation Forest behavioural anomaly signal
        |
        v
Explainable 0–100 Risk Score
        |
        v
VERIFIED / REQUIRES REVIEW / HIGH RISK

SQLite demo history <--- seeded from data/synthetic_invoices.csv
```

The deployment is intentionally a single Streamlit service to reduce operational risk within a 24-hour challenge while still demonstrating data science, explainability, testing, and persistence through a seeded SQLite database.

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run Locally

```bash
streamlit run app.py
```

Run tests:

```bash
PYTHONPATH=. pytest -q
```

Regenerate synthetic data:

```bash
PYTHONPATH=. python src/data_generation.py
```

Regenerate evaluation metrics:

```bash
PYTHONPATH=. python src/evaluate.py
```

## 4. How the ML / Risk Model Works

The project uses a **hybrid detection engine** rather than a purely supervised classifier.

Known financial-control violations are represented as deterministic rules because they are directly auditable and should not depend on a statistical model. Examples include duplicate invoice numbers, an established vendor changing bank account, a 50% VAT rate in the configured demonstration rules, or a missing PO for a high-value invoice.

An **Isolation Forest** model provides an auxiliary unsupervised anomaly signal across behavioural features. This is useful because realistic labelled invoice-fraud data is not available in this challenge. The ML contribution is deliberately bounded so an opaque anomaly model cannot override clear finance controls.

Final score bands:

| Risk score | Business state | Recommendation |
|---:|---|---|
| 0–34 | VERIFIED | No significant indicators detected |
| 35–69 | REQUIRES REVIEW | Review supporting documents |
| 70–100 | HIGH RISK | Hold payment and manually verify |

Critical controls use score floors: confirmed duplicate invoices and established-vendor IBAN changes are forced into `HIGH RISK`.

## 5. Features Used for Fraud Detection

The engine uses current invoice fields plus historical vendor context:

- normalised vendor and invoice number
- duplicate invoice number for the same vendor
- vendor historical median gross amount
- current amount / vendor median ratio
- VAT rate = VAT / net amount
- gross reconciliation: net + VAT ≈ gross
- purchase order presence and amount-sensitive PO risk
- historical/established IBAN comparison
- days since previous vendor invoice
- invoice count in recent windows
- vendor history count
- payment-term days
- logarithm of gross amount
- Isolation Forest decision function

All vendor-relative comparisons use invoice history rather than treating an invoice as an isolated row.

## 6. Dataset Generation

`src/data_generation.py` generates synthetic invoice history with persistent vendor profiles. The repository currently includes **189 invoice records**, exceeding the requested minimum of 100.

The synthetic data contains:

- normal invoices
- duplicates
- extreme amounts
- abnormal vendor frequency
- VAT anomalies
- missing PO examples
- changed IBAN examples
- repeated and new vendors
- EUR, USD, and GBP invoices

No confidential customer/company data is used.

The professor's required baseline is explicitly included:

```text
Vendor: ABC GmbH
Invoice: INV-10001
Net: 1000 EUR
VAT: 190 EUR
Gross: 1190 EUR
PO: PO-5001
IBAN: DE89370400440532013000
```

## 7. Model Selection

### Why not train a Random Forest on synthetic fraud labels?

Training a supervised classifier on fraud labels created from the same hand-written rules can produce deceptively high metrics while mostly learning the synthetic generator. That would not demonstrate real-world generalisation.

Instead:

1. **Rules** detect known and explainable finance-control violations.
2. **Isolation Forest** supplies unsupervised behavioural anomaly detection.
3. A transparent risk aggregator converts evidence into CFO-friendly business states.

This design prioritises explainability and scientific honesty over an artificially impressive synthetic accuracy score.

## 8. Evaluation Methodology

`src/evaluate.py` evaluates injected synthetic anomalies plus normal re-submissions that use new invoice numbers to avoid trivial duplicate leakage.

Current synthetic scenario metrics (`models/evaluation.json`):

- Precision: **1.000**
- Recall: **0.625**
- F1-score: **0.769**
- Accuracy: **0.956**
- Confusion matrix: `[[60, 0], [3, 5]]`

These values are **not estimates of production fraud-detection performance**. They only demonstrate scenario coverage on generated data.

### False positive vs false negative trade-off

A false positive delays a legitimate supplier payment and consumes reviewer time. A false negative can allow a suspicious or fraudulent payment to proceed. The engine therefore intentionally prioritises recall for high-consequence deterministic controls such as duplicates and vendor bank-account changes, while keeping less severe indicators in `REQUIRES REVIEW`.

## 9. API Endpoints

This 24-hour submission uses **Streamlit as the application layer** and does not expose a separate REST API. The challenge states the API endpoints are required *if a backend API is built*. The same risk engine is isolated in `src/fraud_engine.py`, so adding FastAPI later is straightforward without changing model logic.

A production extension would expose:

- `GET /health`
- `POST /analyze`
- `GET /invoices`
- `GET /invoice/{id}`


No API keys or secrets are required for this demo.

The SQLite database is automatically seeded from the included CSV when the app starts. On free ephemeral hosting, new user-entered rows are not intended as permanent production storage; the supplied seed dataset remains reproducible.

## 11. Known Limitations

- The dataset is synthetic and anomalies are deliberately over-represented.
- Synthetic evaluation demonstrates scenario coverage, not production generalisation.
- VAT checking uses simplified demonstration rates rather than a jurisdiction/product tax engine.
- Currency values are compared within same vendor/currency history; there is no FX conversion.
- SQLite persistence on Streamlit Cloud is not production-grade durable storage.
- No OCR/PDF invoice extraction is included.
- No ERP/vendor-master or purchase-order system integration exists.
- Isolation Forest is trained on a small synthetic history.
- Risk score is heuristic + anomaly evidence and is not calibrated as fraud probability.

## 12. Future Improvements

- FastAPI service exposing the required REST endpoints
- managed PostgreSQL database
- ERP/vendor-master integration
- three-way PO / goods receipt / invoice matching
- OCR and structured extraction from PDF/image invoices
- IBAN ownership verification
- country-specific VAT rule service
- near-duplicate invoice matching
- human-review feedback loop and supervised learning with genuine labels
- model calibration, drift monitoring, and versioning
- role-based access control and audit logging

## Required Professor Test Cases

The automated test suite in `tests/test_required_cases.py` covers all five required cases.

### Test 1 – Normal Invoice

Use preset **Normal**. Expected: `VERIFIED`.

### Test 2 – Duplicate

Use preset **Duplicate** (`ABC GmbH`, `INV-10001`). Expected: `HIGH RISK`, duplicate reason.

### Test 3 – Extreme Amount

Use preset **Extreme Amount** (`25,000 EUR`). Expected: `REQUIRES REVIEW` or `HIGH RISK` with vendor-relative amount explanation.

### Test 4 – Changed IBAN

Use preset **Changed IBAN**. Expected: `HIGH RISK`, bank-account-change explanation.

### Test 5 – VAT Inconsistency

Use preset **VAT Inconsistency** (`Net 1000`, `VAT 500`, `Gross 1500`). Expected: `REQUIRES REVIEW` or `HIGH RISK`, VAT inconsistency explanation.

## Repository Structure

```text
invoice_ai_streamlit/
├── app.py
├── README.md
├── TECHNICAL_SUMMARY.md
├── DEPLOYMENT.md
├── requirements.txt
├── .env.example
├── src/
│   ├── fraud_engine.py
│   ├── database.py
│   ├── data_generation.py
│   └── evaluate.py
├── models/
│   └── evaluation.json
├── data/
│   ├── synthetic_invoices.csv
│   ├── sample_upload.csv
│   └── invoice_ai.db  # generated automatically when app runs
└── tests/
    ├── test_required_cases.py
    └── test_engine.py
```

## Security / Secrets

No secrets are required. `.env`, Streamlit secrets, virtual environments, caches, and Python bytecode are excluded via `.gitignore`.
