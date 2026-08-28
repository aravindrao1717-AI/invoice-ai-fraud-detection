from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import json
import pandas as pd
import streamlit as st

from src.fraud_engine import FraudEngine
from src.database import initialize_database, load_invoices

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_invoices.csv"
DB_PATH = ROOT / "data" / "invoice_ai.db"
initialize_database(DB_PATH, DATA_PATH)

st.set_page_config(page_title="Invoice AI – Fraud Detection", page_icon="🧾", layout="wide")

@st.cache_data
def load_data():
    return load_invoices(DB_PATH)

if "history" not in st.session_state:
    st.session_state.history = load_data()

engine = FraudEngine(st.session_state.history)

st.title("🧾 Invoice AI – Intelligent Invoice Fraud Detection")
st.caption("Explainable hybrid risk detection • Synthetic demo data only")

page = st.sidebar.radio("Navigation", ["Dashboard", "Analyze Invoice", "CSV Upload", "Invoice History", "Model & Evaluation"])


def dashboard():
    hist = st.session_state.history.copy()
    assessed = hist[hist.get("risk_level", pd.Series(index=hist.index, dtype=str)).notna()] if "risk_level" in hist.columns else pd.DataFrame()
    if assessed.empty:
        # Score a sample of history for meaningful dashboard values without mutating dataset.
        scores = []
        temp_engine = FraudEngine(hist)
        for i, row in hist.tail(min(100, len(hist))).iterrows():
            result = temp_engine.analyze(row.to_dict())
            scores.append(result.to_dict())
        assessed = pd.DataFrame(scores)
    total = len(hist)
    high = int((assessed["risk_level"] == "HIGH RISK").sum())
    review = int((assessed["risk_level"] == "REQUIRES REVIEW").sum())
    verified = int((assessed["risk_level"] == "VERIFIED").sum())
    risk_rate = (high + review) / max(len(assessed), 1) * 100
    avg_score = assessed["risk_score"].mean() if len(assessed) else 0
    cols = st.columns(6)
    values = [("Invoices Analyzed", total), ("High Risk", high), ("Requires Review", review), ("Verified", verified), ("Risk Rate", f"{risk_rate:.1f}%"), ("Avg Risk Score", f"{avg_score:.1f}/100")]
    for c, (label, value) in zip(cols, values): c.metric(label, value)
    st.subheader("Synthetic dataset")
    st.dataframe(hist.tail(30), use_container_width=True, hide_index=True)


def show_result(result):
    if result.risk_level == "HIGH RISK": st.error(f"HIGH RISK — {result.risk_score}/100")
    elif result.risk_level == "REQUIRES REVIEW": st.warning(f"REQUIRES REVIEW — {result.risk_score}/100")
    else: st.success(f"VERIFIED — {result.risk_score}/100")
    st.write(f"**Recommendation:** {result.recommendation}")
    st.subheader("Why this invoice was classified this way")
    if result.risk_factors:
        for factor in sorted(result.risk_factors, key=lambda x: x["contribution"], reverse=True):
            st.write(f"**{factor['severity']} · {factor['code']} (+{factor['contribution']})** — {factor['message']}")
    else:
        st.write("No significant risk factors detected.")
    with st.expander("Technical details"):
        st.json(result.features)


def analyze_page():
    st.subheader("Manual invoice entry")
    st.info("Tip: use the preset buttons to demonstrate the professor's required test cases.")
    presets = st.columns(5)
    if "preset" not in st.session_state: st.session_state.preset = "Normal"
    names = ["Normal", "Duplicate", "Extreme Amount", "Changed IBAN", "VAT Inconsistency"]
    for c, n in zip(presets, names):
        if c.button(n, use_container_width=True): st.session_state.preset = n
    p = st.session_state.preset
    defaults = {
        "Normal": dict(invoice="INV-NEW-NORMAL", net=1000.0, vat=190.0, gross=1190.0, iban="DE89370400440532013000", po="PO-5001"),
        "Duplicate": dict(invoice="INV-10001", net=1000.0, vat=190.0, gross=1190.0, iban="DE89370400440532013000", po="PO-5001"),
        "Extreme Amount": dict(invoice="INV-EXT-DEMO", net=21008.4, vat=3991.6, gross=25000.0, iban="DE89370400440532013000", po="PO-7001"),
        "Changed IBAN": dict(invoice="INV-IBAN-DEMO", net=1000.0, vat=190.0, gross=1190.0, iban="DE12345678901234567890", po="PO-5001"),
        "VAT Inconsistency": dict(invoice="INV-VAT-DEMO", net=1000.0, vat=500.0, gross=1500.0, iban="DE89370400440532013000", po="PO-5001"),
    }[p]
    with st.form("invoice_form"):
        c1, c2 = st.columns(2)
        vendor = c1.text_input("Vendor", "ABC GmbH", key=f"vendor_{p}")
        invoice_number = c2.text_input("Invoice number", defaults["invoice"], key=f"invoice_{p}")
        invoice_date = c1.date_input("Invoice date", date(2026, 8, 25), key=f"date_{p}")
        due_date = c2.date_input("Due date", date(2026, 9, 24), key=f"due_{p}")
        net = c1.number_input("Net amount", min_value=0.0, value=defaults["net"], step=10.0, key=f"net_{p}")
        vat = c2.number_input("VAT", min_value=0.0, value=defaults["vat"], step=10.0, key=f"vat_{p}")
        gross = c1.number_input("Gross amount", min_value=0.0, value=defaults["gross"], step=10.0, key=f"gross_{p}")
        currency = c2.selectbox("Currency", ["EUR", "USD", "GBP"], index=0, key=f"currency_{p}")
        po = c1.text_input("Purchase order number", defaults["po"], key=f"po_{p}")
        iban = c2.text_input("Bank account / IBAN", defaults["iban"], key=f"iban_{p}")
        terms = c1.text_input("Payment terms", "30 days", key=f"terms_{p}")
        submitted = st.form_submit_button("Analyze Invoice", type="primary", use_container_width=True)
    if submitted:
        payload = {"vendor": vendor, "invoice_number": invoice_number, "invoice_date": str(invoice_date), "due_date": str(due_date), "net_amount": net, "vat_amount": vat, "gross_amount": gross, "currency": currency, "po_number": po, "iban": iban, "payment_terms": terms}
        result = engine.analyze(payload)
        show_result(result)


def csv_page():
    st.subheader("CSV batch analysis")
    uploaded = st.file_uploader("Upload invoice CSV", type=["csv"])
    st.download_button("Download sample CSV", DATA_PATH.read_bytes(), file_name="synthetic_invoices.csv", mime="text/csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview")
        st.dataframe(df.head(20), use_container_width=True)
        required = {"vendor","invoice_number","invoice_date","net_amount","vat_amount","gross_amount","currency","po_number","iban","payment_terms"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            return
        if st.button("Analyze uploaded invoices", type="primary"):
            outputs = []
            temp = FraudEngine(st.session_state.history)
            for _, row in df.iterrows():
                r = temp.analyze(row.to_dict())
                outputs.append({**row.to_dict(), "risk_score": r.risk_score, "risk_level": r.risk_level, "recommendation": r.recommendation, "risk_factors": "; ".join(f["message"] for f in r.risk_factors)})
            out = pd.DataFrame(outputs)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download assessed CSV", out.to_csv(index=False).encode(), file_name="assessed_invoices.csv", mime="text/csv")


def history_page():
    st.subheader("Invoice history")
    df = st.session_state.history
    risk_filter = st.multiselect("Synthetic anomaly type", sorted(df["synthetic_anomaly_types"].astype(str).unique())) if "synthetic_anomaly_types" in df.columns else []
    shown = df[df["synthetic_anomaly_types"].isin(risk_filter)] if risk_filter else df
    st.dataframe(shown, use_container_width=True, hide_index=True)


def model_page():
    st.subheader("Model & evaluation")
    st.markdown("""
**Selected approach:** hybrid rule-based controls + Isolation Forest anomaly detection.

Known controls such as duplicate invoice numbers, changed IBANs, VAT inconsistencies, and missing purchase orders are deterministic and directly explainable. Isolation Forest is used only as a bounded auxiliary signal for unusual behavioural combinations. The displayed value is a **risk score**, not a calibrated fraud probability.
""")
    metrics_path = ROOT / "models" / "evaluation.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Precision", f"{metrics['precision']:.3f}")
        c2.metric("Recall", f"{metrics['recall']:.3f}")
        c3.metric("F1", f"{metrics['f1']:.3f}")
        c4.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.write("Confusion matrix", metrics["confusion_matrix"])
    st.caption("Metrics use synthetic anomaly labels and therefore demonstrate scenario coverage, not real-world fraud prevalence or generalization.")

if page == "Dashboard": dashboard()
elif page == "Analyze Invoice": analyze_page()
elif page == "CSV Upload": csv_page()
elif page == "Invoice History": history_page()
else: model_page()
