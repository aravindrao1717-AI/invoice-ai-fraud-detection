# Invoice AI – Technical Summary

## Approach

I implemented an explainable hybrid invoice-risk application using Streamlit, Pandas, NumPy, scikit-learn, and SQLite. The central design decision was to separate **known financial controls** from **unknown behavioural anomalies**. Deterministic controls handle invoice conditions that should be directly auditable, while Isolation Forest supplies an auxiliary unsupervised signal when the overall invoice behaviour differs from historical patterns.

The final output is intentionally expressed as a **0–100 risk score** and three business states — `VERIFIED`, `REQUIRES REVIEW`, and `HIGH RISK` — rather than as a fraud probability. The score combines evidence from duplicate detection, vendor-relative amount anomalies, VAT checks, missing purchase orders, bank-account changes, unusual invoice frequency, limited vendor history, and a bounded ML anomaly contribution. Duplicate invoices and established-vendor IBAN changes use score floors so they reliably produce high-risk outcomes.

## Dataset

A reproducible synthetic dataset generator creates vendor-specific invoice histories. Vendors have characteristic invoice amounts, invoice cadence, VAT behaviour, currency, IBAN, payment terms, and purchase orders. The included dataset contains 189 records, exceeding the challenge minimum of 100, and contains normal invoices plus deliberately injected duplicate, amount, frequency, VAT, missing-PO, and IBAN-change scenarios. EUR, USD, and GBP examples are present. No confidential or real customer data is used.

The five mandatory professor test cases are explicitly represented and also covered by automated tests. This avoids relying on random generation to accidentally create a valid demo scenario.

## Model and Feature Engineering

Known finance controls are implemented as explicit rules because they provide stronger local explainability than an opaque classifier. For example, duplicate detection compares normalised vendor and invoice number against historical records, while bank-account-change detection compares the submitted IBAN with the established historical IBAN for that vendor.

Amount anomalies are vendor-relative rather than based on a single global threshold. The engine calculates the vendor historical median and the current-to-median ratio. This makes a 25,000 EUR invoice meaningful in the context of a vendor that normally invoices around 1,000 EUR. VAT checks calculate the observed VAT rate and reconcile gross against net plus VAT. Missing PO severity is amount-sensitive, so a high-value invoice without a PO receives more risk than a small one.

Isolation Forest uses behavioural features including log gross amount, amount-to-median ratio, VAT rate, days since previous invoice, seven-day invoice count, vendor history count, and payment-term days. Its contribution is capped so that the statistical model supplements rather than overrides direct financial controls.

## Results and Evaluation

Evaluation is performed on synthetic anomaly scenarios together with normal re-submissions using fresh invoice numbers, preventing duplicate detection from trivially flagging normal historical rows. Current scenario metrics are precision 1.000, recall 0.625, F1 0.769, and accuracy 0.956 with confusion matrix `[[60, 0], [3, 5]]`.

These metrics should not be interpreted as production fraud-detection performance. The dataset is synthetic, anomaly prevalence is artificial, and the patterns were deliberately injected. The purpose of the metrics is to show implementation correctness and scenario coverage.

The false-positive/false-negative trade-off is handled conservatively. A false positive causes review effort and possible supplier-payment delay; a false negative can allow a suspicious payment to proceed. Therefore high-consequence controls such as duplicate invoices and established-vendor bank-account changes are tuned for strong recall, while less decisive signals route invoices to manual review rather than automatically labelling them fraudulent.

## Architecture

The Streamlit UI contains a dashboard, manual invoice form, CSV batch uploader, invoice history, and model/evaluation page. The risk logic is isolated in `src/fraud_engine.py`, data generation in `src/data_generation.py`, evaluation in `src/evaluate.py`, and SQLite seeding/loading in `src/database.py`. The SQLite demo database is initialised from the synthetic CSV to make the application immediately usable after deployment.

This single-service design was selected to maximise delivery reliability within 24 hours. A future production architecture would expose the same engine through FastAPI and persist invoice history in managed PostgreSQL.

## Limitations

The largest limitation is synthetic data. The VAT rules are simplified and are not a complete tax-compliance engine. Currency comparisons do not perform FX conversion. SQLite on Streamlit Community Cloud is not durable production storage. The system does not ingest PDF/image invoices, perform OCR, verify IBAN ownership, integrate with vendor-master or ERP systems, or learn from reviewer outcomes.

A production roadmap would add managed PostgreSQL, FastAPI endpoints, ERP and purchase-order matching, document extraction, country-specific tax logic, bank-account verification, human feedback, supervised fraud learning once genuine labels exist, probability calibration, drift monitoring, rule/model versioning, authentication, and audit logs.
