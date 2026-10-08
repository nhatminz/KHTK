"""Explicit prediction-time feature policy and deterministic transformations."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

TARGETS = ["y1_man_hinh_gop_phan_thieu_dien", "y2_can_nang_cap_may_phat"]
IDENTIFIERS = ["scenario_id", "vehicle_id", "profile_id"]
SAFE_FEATURES = [
    "may_phat_dinh_muc_A_GIA_LAP", "suc_khoe_may_phat_pct_GIA_LAP",
    "suc_khoe_acquy_pct_GIA_LAP", "muc_sac_acquy_SOC_pct_GIA_LAP",
    "ty_le_dung_cho_no_may_do_thi", "vong_tua_may_rpm",
    "tai_dien_co_ban_A", "den_pha_A", "quat_gio_dieu_hoa_A", "suoi_kinh_A",
    "dong_sac_acquy_A", "dau_CD_cu_A", "man_hinh_android_ranh_A",
    "am_luong_pct", "do_nang_luong_am_thanh_RMS_0_1", "amp_ngoai_co_lap",
    "amp_ngoai_dinh_muc_dong_A_GIA_LAP",
]
MEASURED_FEATURES = [
    "amp_ngoai_dong_thuc_A_GIA_LAP", "tai_man_hinh_luc_phat_A_GIA_LAP",
    "loi_lap_dat_dien_GIA_LAP", "tai_roi_nguon_do_loi_A", "sut_ap_mat_V_GIA_LAP",
]
LEAKAGE_FEATURES = [
    "du_dia_dien_moi_A_GIA_LAP", "du_dia_dien_dau_cu_A_GIA_LAP",
    "tong_tai_dien_moi_A_GIA_LAP", "dong_may_phat_co_san_A_GIA_LAP",
    "dien_ap_khi_no_may_V_GIA_LAP", "chet_may_GIA_LAP",
]
MUSIC_FEATURES = ["the_loai_nhac", "ca_si_minh_hoa"]
CONTEXT_FEATURES = ["hang_xe", "mau_xe", "doi_xe", "dung_tich_dong_co_L",
                    "xang_do_thi_L_100km", "xang_hon_hop_L_100km", "toc_do_xe_kmh"]
CATEGORICAL_FEATURES = ["hang_xe", "mau_xe", *MUSIC_FEATURES,
                        "amp_ngoai_co_lap", "loi_lap_dat_dien_GIA_LAP"]
STRING_COLUMNS = [*IDENTIFIERS, "hang_xe", "mau_xe", *MUSIC_FEATURES]
ALL_COLUMNS = [*IDENTIFIERS, "hang_xe", "mau_xe", "doi_xe", "dung_tich_dong_co_L",
               "xang_do_thi_L_100km", "xang_hon_hop_L_100km", *SAFE_FEATURES,
               "toc_do_xe_kmh", *MEASURED_FEATURES, *LEAKAGE_FEATURES, *MUSIC_FEATURES,
               "loi_bom_xang_GIA_LAP", *TARGETS]
NUMERIC_COLUMNS = [c for c in ALL_COLUMNS if c not in STRING_COLUMNS]
BINARY_COLUMNS = ["amp_ngoai_co_lap", "loi_lap_dat_dien_GIA_LAP",
                  "loi_bom_xang_GIA_LAP", "chet_may_GIA_LAP", *TARGETS]
SIGNED_COLUMNS = ["du_dia_dien_moi_A_GIA_LAP", "du_dia_dien_dau_cu_A_GIA_LAP"]
RANGES = {c: (0, None) for c in NUMERIC_COLUMNS if c not in SIGNED_COLUMNS}
RANGES.update({c: (0, 100) for c in NUMERIC_COLUMNS if "pct" in c})
RANGES.update({"ty_le_dung_cho_no_may_do_thi": (0, 1),
               "do_nang_luong_am_thanh_RMS_0_1": (0, 1)})
RANGES.update({c: (0, 1) for c in BINARY_COLUMNS})
RANGES.update({"vong_tua_may_rpm": (1, None), "may_phat_dinh_muc_A_GIA_LAP": (1, None),
               "dung_tich_dong_co_L": (0.01, None), "doi_xe": (1886, 2100)})

ENGINEERED = {
    "phu_tai_co_ban_tong_A": {
        "formula": "tai_dien_co_ban_A + den_pha_A + quat_gio_dieu_hoa_A + suoi_kinh_A",
        "unit": "A", "assumption": "Additive component loads known before prediction; excludes new audio load and available supply."},
    "volume_RMS_proxy": {"formula": "(am_luong_pct / 100) * do_nang_luong_am_thanh_RMS_0_1",
                         "unit": "dimensionless", "assumption": "Interaction proxy, not a linear electrical power law."},
    "SOH_phu_tai_A": {"formula": "(suc_khoe_acquy_pct_GIA_LAP / 100) * phu_tai_co_ban_tong_A",
                      "unit": "A (interaction)", "assumption": "Battery SOH measured or estimated independently before prediction."},
    "rpm_do_thi_proxy": {"formula": "vong_tua_may_rpm * ty_le_dung_cho_no_may_do_thi",
                         "unit": "rpm (interaction)", "assumption": "Scenario RPM and planned urban idle share are known."},
}


def feature_list(variant="safe", engineered=False):
    variants = {"safe": SAFE_FEATURES, "music": SAFE_FEATURES + MUSIC_FEATURES,
                "context": SAFE_FEATURES + CONTEXT_FEATURES,
                "measured": SAFE_FEATURES + MEASURED_FEATURES}
    if variant not in variants:
        raise ValueError(f"Unknown feature variant: {variant}")
    return list(variants[variant]) + (list(ENGINEERED) if engineered else [])


def invalid_numeric_mask(series, column):
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & (numeric.isna() | ~np.isfinite(numeric))
    lower, upper = RANGES.get(column, (None, None))
    if lower is not None:
        invalid |= numeric < lower
    if upper is not None:
        invalid |= numeric > upper
    if column in BINARY_COLUMNS:
        invalid |= numeric.notna() & ~numeric.isin([0, 1])
    return invalid


def categorical_value(value, column):
    """Keep binary categories stable when CSV missing values promote int columns to float."""
    if pd.isna(value) or not str(value).strip():
        return np.nan
    return str(int(value)) if column in BINARY_COLUMNS else str(value)


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Whitelist raw inputs; enforce schema; derive features without fitting statistics."""

    def __init__(self, variant="safe", engineered=False):
        self.variant = variant
        self.engineered = engineered

    def fit(self, X, y=None):
        self.transform_input(X)
        self.feature_names_in_ = np.asarray(feature_list(self.variant), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame with named columns.")
        required = feature_list(self.variant)
        missing = sorted(set(required) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        out = X.loc[:, required].copy()
        for c in required:
            if c in NUMERIC_COLUMNS:
                if invalid_numeric_mask(out[c], c).any():
                    raise ValueError(f"Invalid numeric values or range in feature: {c}")
                out[c] = pd.to_numeric(out[c], errors="raise")
            if c in CATEGORICAL_FEATURES:
                out[c] = out[c].map(lambda v: categorical_value(v, c))
        return out

    def transform(self, X):
        check_is_fitted(self)
        out = self.transform_input(X)
        if self.engineered:
            # Ordinary addition propagates missing components; impute inside training folds later.
            out["phu_tai_co_ban_tong_A"] = (out["tai_dien_co_ban_A"] + out["den_pha_A"]
                + out["quat_gio_dieu_hoa_A"] + out["suoi_kinh_A"])
            out["volume_RMS_proxy"] = out["am_luong_pct"] / 100 * out["do_nang_luong_am_thanh_RMS_0_1"]
            out["SOH_phu_tai_A"] = out["suc_khoe_acquy_pct_GIA_LAP"] / 100 * out["phu_tai_co_ban_tong_A"]
            out["rpm_do_thi_proxy"] = out["vong_tua_may_rpm"] * out["ty_le_dung_cho_no_may_do_thi"]
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(feature_list(self.variant, self.engineered), dtype=object)


def feature_policy(columns):
    """Answer availability, provenance, label-rule use and shortcut risk for every column."""
    records = []
    for c in columns:
        if c in TARGETS:
            role, timing, provenance, rule, risk = ("target", "After simulation", "Rule-generated label", "Is the label", "Never an input")
        elif c in IDENTIFIERS:
            role, timing, provenance, rule, risk = ("identifier", "Before prediction", "Synthetic ID", "No physical role", "Memorization; vehicle_id used only for split, profile_id for audit")
        elif c in LEAKAGE_FEATURES:
            role, timing, provenance, rule, risk = ("excluded_leakage", "After simulated electrical response", "Derived balance/voltage or outcome", "Balance columns directly encode label mechanism; exact generator not supplied", "Direct leakage or downstream shortcut")
        elif c in MEASURED_FEATURES:
            role, timing, provenance, rule, risk = ("measured_only", "Only after independent current/installation/ground inspection BEFORE label decision", "Simulated operating current or diagnostic condition", "Likely component/condition of rule; exact generator not supplied", "High shortcut risk; excluded from baseline, separate measured experiment")
        elif c in SAFE_FEATURES:
            role, timing, provenance, rule, risk = ("safe", "Before prediction, conditional on valid scenario measurements/estimates", "Scenario input/component specification; synthetic values", "May be an upstream rule input; exact generator not supplied", "Learns simulation assumptions; not proof of causal or real-world validity")
        elif c in MUSIC_FEATURES:
            role, timing, provenance, rule, risk = ("music_ablation_only", "Before prediction", "Illustrative category", "No justified physical rule dependence", "Spurious correlation; excluded from principal models")
        elif c in CONTEXT_FEATURES:
            role, timing, provenance, rule, risk = ("context_ablation_only", "Before prediction", "Reference specification/context or simulated speed", "No demonstrated independent benefit", "Profile memorization, redundancy or weak physical relevance")
        elif c == "loi_bom_xang_GIA_LAP":
            role, timing, provenance, rule, risk = ("excluded_other_mechanism", "Only after separate diagnostic inspection", "Simulated independent fuel-pump fault", "Workbook describes separate stall mechanism", "Confounds electrical targets with an unrelated diagnosis")
        else:
            role, timing, provenance, rule, risk = ("excluded_unknown", "Unestablished", "Unknown", "Unknown", "Not reviewed; never include implicitly")
        records.append(dict(column=c, role=role, available_before_prediction=timing,
                            provenance=provenance, use_in_label_rule=rule, reason=risk))
    return records
