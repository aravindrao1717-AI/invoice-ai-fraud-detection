from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any
import math
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


RISK_LEVELS = {"VERIFIED", "REQUIRES REVIEW", "HIGH RISK"}


def _norm_vendor(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _norm_invoice(value: str) -> str:
    return str(value or "").strip().upper()


def _norm_iban(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().upper())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


@dataclass
class RiskFactor:
    code: str
    message: str
    contribution: int
    severity: str


@dataclass
class Assessment:
    risk_score: int
    risk_level: str
    recommendation: str
    risk_factors: list[dict[str, Any]]
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FraudEngine:
    """Explainable hybrid invoice risk engine.

    Deterministic controls handle known finance risks. Isolation Forest adds a bounded
    anomaly signal for unusual behavioural combinations. The score is a risk score,
    not a calibrated fraud probability.
    """

    def __init__(self, history: pd.DataFrame):
        self.history = self._prepare(history)
        self.scaler: RobustScaler | None = None
        self.model: IsolationForest | None = None
        self.ml_features = [
            "log_gross_amount",
            "amount_ratio",
            "vat_rate",
            "days_since_previous",
            "count_7d",
            "history_count",
            "payment_term_days",
        ]
        self._fit_anomaly_model()

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return df
        for col in ["vendor", "invoice_number", "iban"]:
            if col not in df.columns:
                df[col] = ""
        df["vendor_norm"] = df["vendor"].map(_norm_vendor)
        df["invoice_number_norm"] = df["invoice_number"].map(_norm_invoice)
        df["iban_norm"] = df["iban"].map(_norm_iban)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        for col in ["net_amount", "vat_amount", "gross_amount"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if "payment_terms" not in df.columns:
            df["payment_terms"] = "30 days"
        return df.sort_values("invoice_date").reset_index(drop=True)

    @staticmethod
    def _payment_days(value: Any) -> int:
        m = re.search(r"(\d+)", str(value or ""))
        return int(m.group(1)) if m else 30

    def _feature_vector(self, invoice: dict[str, Any], prior: pd.DataFrame) -> dict[str, float]:
        gross = max(_safe_float(invoice.get("gross_amount")), 0.0)
        net = _safe_float(invoice.get("net_amount"))
        vat = _safe_float(invoice.get("vat_amount"))
        vendor = _norm_vendor(invoice.get("vendor", ""))
        currency = str(invoice.get("currency", "EUR")).upper()
        vendor_hist = prior[(prior["vendor_norm"] == vendor) & (prior["currency"].str.upper() == currency)] if not prior.empty else prior
        history_count = int(len(vendor_hist))
        median = float(vendor_hist["gross_amount"].median()) if history_count else gross or 1.0
        amount_ratio = gross / median if median > 0 else 1.0
        inv_date = pd.to_datetime(invoice.get("invoice_date"), errors="coerce")
        if history_count and pd.notna(inv_date):
            previous_dates = vendor_hist["invoice_date"].dropna()
            previous_dates = previous_dates[previous_dates < inv_date]
            days_since_previous = int((inv_date - previous_dates.max()).days) if len(previous_dates) else 90
            count_7d = int(((inv_date - vendor_hist["invoice_date"]).dt.days.between(0, 7)).sum())
        else:
            days_since_previous = 90
            count_7d = 0
        return {
            "log_gross_amount": float(math.log1p(gross)),
            "amount_ratio": float(amount_ratio),
            "vat_rate": float(vat / net if net > 0 else 0.0),
            "days_since_previous": float(days_since_previous),
            "count_7d": float(count_7d),
            "history_count": float(history_count),
            "payment_term_days": float(self._payment_days(invoice.get("payment_terms"))),
        }

    def _fit_anomaly_model(self) -> None:
        if len(self.history) < 30:
            return
        rows = []
        for idx, row in self.history.iterrows():
            prior = self.history.iloc[:idx]
            rows.append(self._feature_vector(row.to_dict(), prior))
        X = pd.DataFrame(rows)[self.ml_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = IsolationForest(n_estimators=150, contamination=0.08, random_state=42)
        self.model.fit(Xs)

    def analyze(self, invoice: dict[str, Any], persist: bool = False) -> Assessment:
        row = dict(invoice)
        row["vendor_norm"] = _norm_vendor(row.get("vendor", ""))
        row["invoice_number_norm"] = _norm_invoice(row.get("invoice_number", ""))
        row["iban_norm"] = _norm_iban(row.get("iban", ""))
        inv_date = pd.to_datetime(row.get("invoice_date"), errors="coerce")
        prior = self.history[self.history["invoice_date"] <= inv_date] if (not self.history.empty and pd.notna(inv_date)) else self.history

        factors: list[RiskFactor] = []
        score = 0
        score_floor = 0
        vendor_hist = prior[prior["vendor_norm"] == row["vendor_norm"]] if not prior.empty else prior

        # Duplicate invoice number for same vendor.
        duplicate = False
        if not vendor_hist.empty and row["invoice_number_norm"]:
            duplicate = bool((vendor_hist["invoice_number_norm"] == row["invoice_number_norm"]).any())
        if duplicate:
            factors.append(RiskFactor("DUPLICATE_INVOICE", "Duplicate invoice detected for this vendor and invoice number.", 70, "CRITICAL"))
            score += 70
            score_floor = max(score_floor, 85)

        # Amount anomaly against same vendor + currency.
        currency = str(row.get("currency", "EUR")).upper()
        same_ccy = vendor_hist[vendor_hist["currency"].str.upper() == currency] if not vendor_hist.empty else vendor_hist
        gross = _safe_float(row.get("gross_amount"))
        hist_count = len(same_ccy)
        median_amount = float(same_ccy["gross_amount"].median()) if hist_count else 0.0
        amount_ratio = gross / median_amount if median_amount > 0 else 1.0
        if hist_count >= 3:
            if amount_ratio >= 10:
                pts = 45
            elif amount_ratio >= 4:
                pts = 30
            elif amount_ratio >= 2.5:
                pts = 15
            else:
                pts = 0
            if pts:
                factors.append(RiskFactor("UNUSUAL_AMOUNT", f"Invoice amount is {amount_ratio:.1f}× the vendor's historical median ({median_amount:,.2f} {currency}).", pts, "HIGH" if pts >= 30 else "MEDIUM"))
                score += pts

        # VAT anomaly.
        net = _safe_float(row.get("net_amount"))
        vat = _safe_float(row.get("vat_amount"))
        vat_rate = vat / net if net > 0 else 0.0
        gross_expected = net + vat
        if abs(gross - gross_expected) > max(1.0, 0.01 * max(gross, 1.0)):
            factors.append(RiskFactor("TOTAL_MISMATCH", "Gross amount does not reconcile with net amount plus VAT.", 25, "HIGH"))
            score += 25
        expected_rates = [0.0, 0.07, 0.19]
        if net > 0 and min(abs(vat_rate - r) for r in expected_rates) > 0.035:
            factors.append(RiskFactor("VAT_ANOMALY", f"VAT rate is {vat_rate*100:.1f}%, inconsistent with configured demonstration tax rates.", 40, "HIGH"))
            score += 40

        # Missing PO with amount-sensitive severity.
        po = str(row.get("po_number", "") or "").strip()
        if not po:
            if gross >= 25000:
                pts = 25
            elif gross >= 10000:
                pts = 20
            else:
                pts = 5
            factors.append(RiskFactor("MISSING_PO", "Purchase order is missing" + (" for a high-value invoice." if gross >= 10000 else "."), pts, "MEDIUM" if pts < 25 else "HIGH"))
            score += pts

        # IBAN change for an established vendor.
        iban_changed = False
        if len(vendor_hist) >= 2 and row["iban_norm"]:
            known = vendor_hist["iban_norm"].replace("", np.nan).dropna()
            if len(known):
                established = str(known.mode().iloc[0])
                if row["iban_norm"] != established:
                    iban_changed = True
                    factors.append(RiskFactor("IBAN_CHANGE", "Vendor bank account differs from the established historical IBAN.", 55, "CRITICAL"))
                    score += 55
                    score_floor = max(score_floor, 80)

        # Frequency anomaly.
        if len(vendor_hist) >= 3 and pd.notna(inv_date):
            dates = vendor_hist["invoice_date"].dropna().sort_values()
            recent_2d = int(((inv_date - dates).dt.days.between(0, 2)).sum())
            gaps = dates.diff().dt.days.dropna()
            median_gap = float(gaps.median()) if len(gaps) else 30.0
            if recent_2d >= 3 and median_gap >= 7:
                factors.append(RiskFactor("UNUSUAL_FREQUENCY", f"Vendor has {recent_2d} invoices within 2 days versus a historical median gap of {median_gap:.0f} days.", 20, "MEDIUM"))
                score += 20

        # Limited vendor history.
        if len(vendor_hist) == 0:
            factors.append(RiskFactor("NEW_VENDOR", "New vendor: no historical invoices are available for behavioural comparison.", 10, "LOW"))
            score += 10

        # Bounded ML signal.
        features = self._feature_vector(row, prior)
        ml_anomaly_raw = None
        if self.model is not None and self.scaler is not None:
            X = pd.DataFrame([[features[c] for c in self.ml_features]], columns=self.ml_features)
            Xs = self.scaler.transform(X)
            decision = float(self.model.decision_function(Xs)[0])
            ml_anomaly_raw = decision
            if decision < -0.10:
                factors.append(RiskFactor("ML_ANOMALY", "Behavioural anomaly model found a strong deviation from historical invoice patterns.", 20, "MEDIUM"))
                score += 20
            elif decision < 0.0:
                factors.append(RiskFactor("ML_ANOMALY", "Behavioural anomaly model found a moderate deviation from historical invoice patterns.", 10, "LOW"))
                score += 10
        features["ml_decision_function"] = ml_anomaly_raw
        features["vendor_median_amount"] = round(median_amount, 2)
        features["amount_ratio_display"] = round(amount_ratio, 2)
        features["duplicate"] = duplicate
        features["iban_changed"] = iban_changed

        score = int(min(100, max(score, score_floor)))
        if score >= 70:
            level = "HIGH RISK"
            recommendation = "MANUAL REVIEW REQUIRED — hold payment until verified."
        elif score >= 35:
            level = "REQUIRES REVIEW"
            recommendation = "Review supporting documents before approval."
        else:
            level = "VERIFIED"
            recommendation = "No significant risk indicators detected."

        assessment = Assessment(score, level, recommendation, [asdict(f) for f in factors], features)
        if persist:
            persisted = {k: row.get(k, "") for k in [
                "vendor", "invoice_number", "invoice_date", "due_date", "net_amount", "vat_amount", "gross_amount", "currency", "po_number", "iban", "payment_terms"
            ]}
            persisted.update({"risk_score": score, "risk_level": level})
            self.history = self._prepare(pd.concat([self.history, pd.DataFrame([persisted])], ignore_index=True))
        return assessment
