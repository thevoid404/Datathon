"""
REACT 2026 Datathon - LEGAL EXTREME PROBE.

Rules-safe aggressive experiment: no external data, no manual labels, no test labels,
no test rows enter train/validation fitting, and target encodings use only historical
TRAIN labels. Test rows are used only at final inference; their fraud column is NaN.
This is deliberately high-variance and is NOT the recommended final submission.

The stable submission scored 0.55478 against 0.55503 and 0.55493 at the top - a spread
smaller than the measured noise. Grinding that gap is not worth an attempt. This file goes
after a large effect instead, and is expected to either beat the stable model clearly or
lose clearly.

THE BET: lagged target encoding. Historical fraud rate per merchant / device / customer /
location is normally the strongest feature in fraud detection, and the stable model uses
none of it. The cold-start report says why that is expensive here - unseen device 96% fraud,
unseen merchant 34%, against a 1.6% base rate. Entity identity carries enormous signal. The
staleness trap is handled by giving every row, train and test alike, the same 45-day label
lag; see the long comment above add_target_encoding.

THE CORRECTION: recency weighting was rejected earlier at -0.006, but it was measured on a
61-day holdout that was 72% OLD regime. A change that helps the new regime and hurts the old
one scores as a loss there, even though the old regime does not appear in test.csv at all.
The REGIME fold here trains on everything before 2026-06-29 and scores only the new regime,
so recency is re-tested on the right target.

THE BUG FIX: _window_stats separated entity blocks by 10 days while using windows of up to
30, so c_cnt_30d and c_sum_30d were counting rows from the previous entity. Fixed and A/B'd
as 00_base vs 01_spanfix.

Two folds. REGIME (train < 2026-06-29, score 2026-06-29 -> 2026-07-15) decides; it has only
~1,000 positives so its SE is near 0.02 and only large effects are visible. GUARD (the
familiar 61-day window) exists to catch a config that wins the regime fold by breaking the
model generally.

  python aggressive.py           run the sweep
  python aggressive.py submit    write submission_aggressive.csv

Feature engineering is otherwise UNCHANGED.
"""
import glob
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

# --------------------------------------------------------------------------------------
# Paths: works on Kaggle (/kaggle/input/...) and locally (same folder as this file)
# --------------------------------------------------------------------------------------
def find_data_dir():
    candidates = glob.glob("/kaggle/input/**/train.csv", recursive=True) + ["train.csv"]
    for p in candidates:
        if os.path.exists(p):
            return os.path.dirname(p) or "."
    raise FileNotFoundError("train.csv not found - add the competition data to the notebook (Add Input)")


DATA_DIR = find_data_dir()
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

CAT_COLS = ["merchant_category", "device_type", "location", "payment_method", "transaction_type"]
ID_COLS = ["customer_id", "merchant_id", "device_id"]
HOUR = 3600
DAY = 86400

# Gap inserted between adjacent key blocks inside _window_stats. It MUST exceed the largest
# window used, or `comb - window` lands inside the PREVIOUS entity's block and the count
# picks up rows belonging to a different entity. The original value was 10 days while the
# largest window is 30 (c_cnt_30d / c_sum_30d), so those two features were contaminated for
# every transaction in the first ~20 days of an entity's block. Not a leak - the rows are
# still in the past - but wrong. Set by the caller; 100 makes all windows safe.
SPAN_BUFFER_DAYS = 10


# --------------------------------------------------------------------------------------
# Feature helpers (all strictly past-only)  -- UNCHANGED
# --------------------------------------------------------------------------------------
def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s.astype(str), sort=False)[0].astype(np.int64)


def _window_stats(key: np.ndarray, t: np.ndarray, window: int, values=None):
    """
    For each row i: count (and optionally sum of `values`) of rows j with the same key and
    t_i - window <= t_j < t_i  (strictly before i). `key`,`t` must be sorted by (key, t).
    """
    span = np.int64(t.max() - t.min() + SPAN_BUFFER_DAYS * DAY)
    comb = key * span + (t - t.min())
    left = np.searchsorted(comb, comb - window, side="left")
    cur = np.searchsorted(comb, comb, side="left")
    cnt = cur - left
    if values is None:
        return cnt
    cs = np.concatenate([[0.0], np.cumsum(values)])
    return cnt, cs[cur] - cs[left]


def _expanding_past(df_sorted, key, val, prefix):
    """Past-only expanding count / mean / std / max of `val` grouped by `key`."""
    g = df_sorted.groupby(key, sort=False)[val]
    n = g.cumcount().astype(np.float64)
    csum = g.cumsum() - df_sorted[val]
    csq = (df_sorted[val] ** 2).groupby(df_sorted[key], sort=False).cumsum() - df_sorted[val] ** 2
    cmax = g.cummax().groupby(df_sorted[key], sort=False).shift(1)
    mean = csum / n.replace(0, np.nan)
    var = csq / n.replace(0, np.nan) - mean**2
    out = pd.DataFrame(index=df_sorted.index)
    out[f"{prefix}_n"] = n
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = np.sqrt(var.clip(lower=0))
    out[f"{prefix}_max"] = cmax
    return out


def _past_mean(df_sorted, key, val):
    v = val.astype(float).fillna(0)
    n = df_sorted.groupby(key, sort=False).cumcount().astype(float)
    cs = v.groupby(df_sorted[key], sort=False).cumsum() - v
    return cs / n.replace(0, np.nan)


def _entity_profile(df, key, prefix, cols):
    """Behavioral profile of an entity from the anomaly indicators of its PAST transactions."""
    out = pd.DataFrame(index=df.index)
    for name, series in cols.items():
        out[f"{prefix}_hist_{name}"] = _past_mean(df, key, series)
    return out


# --------------------------------------------------------------------------------------
# Feature engineering  -- UNCHANGED
# --------------------------------------------------------------------------------------
def build_features(train, test):
    train = train.copy()
    test = test.copy()
    train["is_test"] = 0
    test["is_test"] = 1
    test["fraud"] = np.nan
    df = pd.concat([train, test], ignore_index=True)

    df["_ord"] = np.arange(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "_ord"], kind="mergesort").reset_index(drop=True)

    ts = df["timestamp"]
    df["t_sec"] = (ts - ts.min()).dt.total_seconds().astype(np.int64)
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_night"] = ((df["hour"] <= 4) | (df["hour"] >= 23)).astype(np.int8)
    df["hour_frac"] = df["hour"] + ts.dt.minute / 60.0
    df["log_amt"] = np.log1p(df["amount_bdt"])
    df["amt_round"] = (df["amount_bdt"] % 100 == 0).astype(np.int8)
    df["amt_round_1000"] = (df["amount_bdt"] % 1000 == 0).astype(np.int8)

    for c in CAT_COLS:
        df[f"{c}_missing"] = df[c].isna().astype(np.int8)
    df["n_missing"] = df[[f"{c}_missing" for c in CAT_COLS]].sum(axis=1)
    for c in ID_COLS + ["location"]:
        df[f"{c}_code"] = _codes(df[c])

    # ---------------- Customer history ----------------
    df = df.sort_values(["customer_id_code", "t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    exp = _expanding_past(df, "customer_id_code", "amount_bdt", "c_amt")
    df = pd.concat([df, exp], axis=1)
    gc = df.groupby("customer_id_code", sort=False)
    df["c_amt_z"] = (df["amount_bdt"] - df["c_amt_mean"]) / (df["c_amt_std"] + 1.0)
    df["c_amt_ratio_mean"] = df["amount_bdt"] / (df["c_amt_mean"] + 1.0)
    df["c_amt_ratio_max"] = df["amount_bdt"] / (df["c_amt_max"] + 1.0)
    df["c_amt_over_max"] = (df["amount_bdt"] > df["c_amt_max"]).astype(np.int8)
    exp_l = _expanding_past(df, "customer_id_code", "log_amt", "c_lamt")
    df["c_lamt_z"] = (df["log_amt"] - exp_l["c_lamt_mean"]) / (exp_l["c_lamt_std"] + 0.1)
    df["c_lamt_mean"] = exp_l["c_lamt_mean"]

    prev_t = gc["t_sec"].shift(1)
    prev_t2 = gc["t_sec"].shift(2)
    df["c_dt_prev"] = df["t_sec"] - prev_t
    df["c_dt_prev2"] = df["t_sec"] - prev_t2
    df["c_log_dt_prev"] = np.log1p(df["c_dt_prev"])
    df["c_first_seen_days"] = (df["t_sec"] - gc["t_sec"].transform("first")) / DAY
    gap = df["c_dt_prev"].fillna(0)
    gap_cum = gap.groupby(df["customer_id_code"], sort=False).cumsum() - gap
    df["c_mean_gap"] = gap_cum / (df["c_amt_n"] - 1).replace(0, np.nan).clip(lower=1)
    df["c_gap_ratio"] = df["c_dt_prev"] / (df["c_mean_gap"] + 60)

    df["c_prev_amt"] = gc["amount_bdt"].shift(1)
    df["c_amt_ratio_prev"] = df["amount_bdt"] / (df["c_prev_amt"] + 1.0)
    df["c_same_amt_prev"] = (df["amount_bdt"] == df["c_prev_amt"]).astype(np.int8)
    df["c_loc_changed"] = (gc["location_code"].shift(1) != df["location_code"]).astype(np.int8)
    df.loc[prev_t.isna(), "c_loc_changed"] = 0
    df["c_dev_changed"] = (gc["device_id_code"].shift(1) != df["device_id_code"]).astype(np.int8)
    df.loc[prev_t.isna(), "c_dev_changed"] = 0
    df["c_merch_changed"] = (gc["merchant_id_code"].shift(1) != df["merchant_id_code"]).astype(np.int8)
    df["c_teleport"] = df["c_loc_changed"] / (np.log1p(df["c_dt_prev"].fillna(1e7)) + 1)
    df["c_prev_night"] = gc["is_night"].shift(1)

    key = df["customer_id_code"].to_numpy()
    t = df["t_sec"].to_numpy()
    amt = df["amount_bdt"].to_numpy()
    for w, name in [(HOUR, "1h"), (6 * HOUR, "6h"), (DAY, "24h"), (7 * DAY, "7d"), (30 * DAY, "30d")]:
        cnt, sm = _window_stats(key, t, w, amt)
        df[f"c_cnt_{name}"] = cnt
        df[f"c_sum_{name}"] = sm
    df["c_amt_share_24h"] = df["amount_bdt"] / (df["c_sum_24h"] + df["amount_bdt"])
    df["c_burst_1h_vs_7d"] = df["c_cnt_1h"] / (df["c_cnt_7d"] / (7 * 24) + 0.05)
    df["c_burst_24h_vs_30d"] = df["c_cnt_24h"] / (df["c_cnt_30d"] / 30 + 0.1)
    df["c_sum24h_vs_mean"] = df["c_sum_24h"] / (df["c_amt_mean"] + 1)

    night_cum = gc["is_night"].cumsum() - df["is_night"]
    df["c_night_frac"] = night_cum / df["c_amt_n"].replace(0, np.nan)
    df["c_night_dev"] = df["is_night"] - df["c_night_frac"].fillna(0.1)
    ang = 2 * np.pi * df["hour_frac"] / 24
    cs_ = np.cos(ang)
    sn_ = np.sin(ang)
    cs_cum = cs_.groupby(df["customer_id_code"], sort=False).cumsum() - cs_
    sn_cum = sn_.groupby(df["customer_id_code"], sort=False).cumsum() - sn_
    n = df["c_amt_n"].replace(0, np.nan)
    mean_ang = np.arctan2(sn_cum / n, cs_cum / n)
    r = np.sqrt((sn_cum / n) ** 2 + (cs_cum / n) ** 2)
    d = np.abs(np.angle(np.exp(1j * (ang - mean_ang))))
    df["c_hour_dev"] = d * r
    df["c_hour_regularity"] = r

    for ent in ["device_id_code", "merchant_id_code", "location_code", "merchant_category", "payment_method", "transaction_type"]:
        pair_n = df.groupby(["customer_id_code", ent], sort=False, dropna=False).cumcount()
        short = ent.replace("_id_code", "").replace("_code", "")
        df[f"c_{short}_pair_n"] = pair_n
        df[f"c_{short}_new"] = (pair_n == 0).astype(np.int8)
        first = (pair_n == 0).astype(np.int64)
        df[f"c_{short}_nuniq"] = first.groupby(df["customer_id_code"], sort=False).cumsum() - first
        df[f"c_{short}_pair_share"] = pair_n / df["c_amt_n"].replace(0, np.nan)
    df["c_new_dev_and_loc"] = df["c_device_new"] * df["c_location_new"]
    df["c_new_dev_and_merch"] = df["c_device_new"] * df["c_merchant_new"]
    df["c_n_new_entities"] = df["c_device_new"] + df["c_location_new"] + df["c_merchant_new"]
    _, s = _window_stats(key, t, 7 * DAY, df["c_device_new"].to_numpy().astype(float))
    df["c_new_dev_7d"] = s

    df["c_prev_amt_ratio"] = gc["c_amt_ratio_mean"].shift(1)
    df["c_prev_dev_new"] = gc["c_device_new"].shift(1)
    df["c_prev_loc_new"] = gc["c_location_new"].shift(1)
    df["c_prev_n_new"] = gc["c_n_new_entities"].shift(1)
    df["c_prev2_n_new"] = gc["c_n_new_entities"].shift(2)
    _, s = _window_stats(key, t, DAY, df["c_amt_ratio_mean"].fillna(1.0).to_numpy())
    df["c_sum_amt_ratio_24h"] = s
    _, s = _window_stats(key, t, 7 * DAY, df["c_n_new_entities"].to_numpy().astype(float))
    df["c_new_entities_7d"] = s
    _, s = _window_stats(key, t, DAY, df["is_night"].to_numpy().astype(float))
    df["c_night_cnt_24h"] = s
    _, s = _window_stats(key, t, DAY, df["c_merchant_new"].to_numpy().astype(float))
    df["c_new_merch_24h"] = s
    _, s = _window_stats(key, t, DAY, df["c_location_new"].to_numpy().astype(float))
    df["c_new_loc_24h"] = s

    first_age = gc["account_age_days"].transform("first")
    df["c_age_expected"] = first_age + np.floor(df["c_first_seen_days"])
    df["c_age_dev"] = df["account_age_days"] - df["c_age_expected"]
    df["c_age_dev_abs"] = df["c_age_dev"].abs()
    df["c_prev_age_dev"] = df["account_age_days"] - gc["account_age_days"].shift(1) - (df["c_dt_prev"] / DAY)

    # ---------------- Device history ----------------
    df = df.sort_values(["device_id_code", "t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    gd = df.groupby("device_id_code", sort=False)
    df["d_n"] = gd.cumcount()
    df["d_dt_prev"] = df["t_sec"] - gd["t_sec"].shift(1)
    df["d_first_seen_days"] = (df["t_sec"] - gd["t_sec"].transform("first")) / DAY
    pair_n = df.groupby(["device_id_code", "customer_id_code"], sort=False).cumcount()
    first = (pair_n == 0).astype(np.int64)
    df["d_cust_nuniq"] = first.groupby(df["device_id_code"], sort=False).cumsum() - first
    df["d_cust_new_on_device"] = first.astype(np.int8)
    df["d_prev_cust_diff"] = (gd["customer_id_code"].shift(1) != df["customer_id_code"]).astype(np.int8)
    df.loc[df["d_n"] == 0, "d_prev_cust_diff"] = 0
    df["d_cust_share"] = pair_n / df["d_n"].replace(0, np.nan)
    key = df["device_id_code"].to_numpy()
    t = df["t_sec"].to_numpy()
    for w, name in [(HOUR, "1h"), (DAY, "24h"), (7 * DAY, "7d")]:
        cnt, sm = _window_stats(key, t, w, df["amount_bdt"].to_numpy())
        df[f"d_cnt_{name}"] = cnt
        df[f"d_sum_{name}"] = sm
    _, s = _window_stats(key, t, 7 * DAY, first.to_numpy().astype(float))
    df["d_new_cust_7d"] = s
    _, s = _window_stats(key, t, DAY, first.to_numpy().astype(float))
    df["d_new_cust_24h"] = s
    dexp = _expanding_past(df, "device_id_code", "amount_bdt", "d_amt")
    df["d_amt_mean"] = dexp["d_amt_mean"]
    df["d_amt_ratio"] = df["amount_bdt"] / (dexp["d_amt_mean"] + 1)
    df["d_burst_24h_vs_7d"] = df["d_cnt_24h"] / (df["d_cnt_7d"] / 7 + 0.1)
    df["d_rate"] = df["d_n"] / (df["d_first_seen_days"] + 1)
    df["d_burst_1h"] = df["d_cnt_1h"] / (df["d_rate"] / 24 + 0.05)
    df["d_new_cust_share_7d"] = df["d_new_cust_7d"] / (df["d_cust_nuniq"] + 1)
    # NOTE: collision-safe pair key (was hard-coded 100000 in the original).
    _cust_mult = np.int64(df["customer_id_code"].max()) + 1
    pair_key = df["device_id_code"].to_numpy() * _cust_mult + df["customer_id_code"].to_numpy()
    order = np.lexsort((df["_ord"].to_numpy(), t, pair_key))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    for w, name in [(DAY, "24h"), (7 * DAY, "7d")]:
        pc = _window_stats(pair_key[order], t[order], w, None)[inv]
        df[f"d_other_cust_cnt_{name}"] = df[f"d_cnt_{name}"] - pc
    df["d_other_cust_frac_24h"] = df["d_other_cust_cnt_24h"] / (df["d_cnt_24h"] + 1)
    aexp = _expanding_past(df, "device_id_code", "account_age_days", "d_age")
    df["d_age_mean"] = aexp["d_age_mean"]
    df["d_age_min_ratio"] = df["account_age_days"] / (aexp["d_age_mean"] + 1)
    dn = gd["is_night"].cumsum() - df["is_night"]
    df["d_night_frac"] = dn / df["d_n"].replace(0, np.nan)
    df["d_merch_pair_n"] = df.groupby(["device_id_code", "merchant_id_code"], sort=False).cumcount()
    first_dm = (df["d_merch_pair_n"] == 0).astype(np.int64)
    df["d_merch_nuniq"] = first_dm.groupby(df["device_id_code"], sort=False).cumsum() - first_dm
    df["d_loc_pair_n"] = df.groupby(["device_id_code", "location_code"], sort=False).cumcount()
    first_dl = (df["d_loc_pair_n"] == 0).astype(np.int64)
    df["d_loc_nuniq"] = first_dl.groupby(df["device_id_code"], sort=False).cumsum() - first_dl
    df["d_loc_new"] = first_dl.astype(np.int8)

    # ---------------- Merchant history ----------------
    df = df.sort_values(["merchant_id_code", "t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    gm = df.groupby("merchant_id_code", sort=False)
    df["m_n"] = gm.cumcount()
    df["m_first_seen_days"] = (df["t_sec"] - gm["t_sec"].transform("first")) / DAY
    df["m_rate"] = df["m_n"] / (df["m_first_seen_days"] + 1)
    mexp = _expanding_past(df, "merchant_id_code", "amount_bdt", "m_amt")
    df["m_amt_mean"] = mexp["m_amt_mean"]
    df["m_amt_std"] = mexp["m_amt_std"]
    df["m_amt_z"] = (df["amount_bdt"] - df["m_amt_mean"]) / (df["m_amt_std"] + 1)
    df["m_amt_ratio"] = df["amount_bdt"] / (df["m_amt_mean"] + 1)
    key = df["merchant_id_code"].to_numpy()
    t = df["t_sec"].to_numpy()
    for w, name in [(HOUR, "1h"), (DAY, "24h"), (7 * DAY, "7d")]:
        cnt, sm = _window_stats(key, t, w, df["amount_bdt"].to_numpy())
        df[f"m_cnt_{name}"] = cnt
        df[f"m_sum_{name}"] = sm
    df["m_burst_1h"] = df["m_cnt_1h"] / (df["m_rate"] / 24 + 0.05)
    df["m_burst_24h"] = df["m_cnt_24h"] / (df["m_rate"] + 0.1)
    pair_n = df.groupby(["merchant_id_code", "customer_id_code"], sort=False).cumcount()
    first = (pair_n == 0).astype(np.int64)
    df["m_cust_nuniq"] = first.groupby(df["merchant_id_code"], sort=False).cumsum() - first
    _, s = _window_stats(key, t, DAY, first.to_numpy().astype(float))
    df["m_new_cust_24h"] = s
    df["m_new_cust_24h_frac"] = df["m_new_cust_24h"] / (df["m_cnt_24h"] + 1)
    mn = gm["is_night"].cumsum() - df["is_night"]
    df["m_night_frac"] = mn / df["m_n"].replace(0, np.nan)
    pair_n = df.groupby(["merchant_id_code", "device_id_code"], sort=False).cumcount()
    df["m_dev_pair_n"] = pair_n
    first_md = (pair_n == 0).astype(np.int64)
    df["m_dev_nuniq"] = first_md.groupby(df["merchant_id_code"], sort=False).cumsum() - first_md
    _, s = _window_stats(key, t, 7 * DAY, first_md.to_numpy().astype(float))
    df["m_new_dev_7d"] = s
    _, s = _window_stats(key, t, 7 * DAY, first.to_numpy().astype(float))
    df["m_new_cust_7d"] = s
    df["m_new_cust_7d_frac"] = df["m_new_cust_7d"] / (df["m_cnt_7d"] + 1)
    df["m_new_cust_7d_vs_hist"] = df["m_new_cust_7d"] / (df["m_cust_nuniq"] / (df["m_first_seen_days"] / 7 + 1) + 1)
    maexp = _expanding_past(df, "merchant_id_code", "account_age_days", "m_age")
    df["m_age_mean"] = maexp["m_age_mean"]
    df["m_age_ratio"] = df["account_age_days"] / (maexp["m_age_mean"] + 1)
    shared = (df["d_cust_nuniq"] >= 1).astype(float).to_numpy()
    _, s = _window_stats(key, t, DAY, shared)
    df["m_shared_dev_frac_24h"] = s / (df["m_cnt_24h"] + 1)
    df["m_cust_per_dev"] = df["m_cust_nuniq"] / (df["m_dev_nuniq"] + 1)

    # ---------------- Location history ----------------
    df = df.sort_values(["location_code", "t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    df["l_n"] = df.groupby("location_code", sort=False).cumcount()
    key = df["location_code"].to_numpy()
    t = df["t_sec"].to_numpy()
    df["l_cnt_1h"] = _window_stats(key, t, HOUR, None)
    df["l_cnt_7d"] = _window_stats(key, t, 7 * DAY, None)
    df["l_burst_1h"] = df["l_cnt_1h"] / (df["l_cnt_7d"] / (7 * 24) + 0.5)
    lexp = _expanding_past(df, "location_code", "amount_bdt", "l_amt")
    df["l_amt_ratio"] = df["amount_bdt"] / (lexp["l_amt_mean"] + 1)

    # ---------------- Entity behavioral profiles (label-free) ----------------
    df["amt_ratio_clip"] = df["c_amt_ratio_mean"].fillna(1.0).clip(upper=20)
    df["log_amt_ratio"] = np.log1p(df["amt_ratio_clip"])
    df["is_big"] = (df["amount_bdt"] > 5000).astype(np.int8)

    df = df.sort_values(["merchant_id_code", "t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    m_gap = df["t_sec"] - df.groupby("merchant_id_code", sort=False)["t_sec"].shift(1)
    df["m_dt_prev"] = m_gap
    df["m_gap_lt_1h"] = (m_gap < HOUR).astype(np.int8)
    df["m_gap_lt_10m"] = (m_gap < 600).astype(np.int8)
    prof = _entity_profile(df, "merchant_id_code", "m", {
        "gap_lt_1h": df["m_gap_lt_1h"], "gap_lt_10m": df["m_gap_lt_10m"],
        "log_gap": np.log1p(m_gap.fillna(30 * DAY)), "night": df["is_night"],
        "amt_ratio": df["log_amt_ratio"], "big": df["is_big"], "cust_new": df["c_merchant_new"],
        "dev_new": df["c_device_new"], "loc_new": df["c_location_new"], "n_new": df["c_n_new_entities"],
        "dev_shared": df["d_cust_nuniq"].clip(upper=30), "cust_hist_n": np.log1p(df["c_amt_n"]),
        "log_amt": df["log_amt"], "age_dev": df["c_age_dev_abs"].clip(upper=30),
        "dev_changed": df["c_dev_changed"], "hour_dev": df["c_hour_dev"].fillna(0),
    })
    df = pd.concat([df, prof], axis=1)
    df["m_amt_ratio_vs_hist"] = df["log_amt_ratio"] - df["m_hist_amt_ratio"]
    key = df["merchant_id_code"].to_numpy()
    t = df["t_sec"].to_numpy()
    df["m_cnt_10m"] = _window_stats(key, t, 600, None)
    _, s = _window_stats(key, t, DAY, df["c_merchant_new"].to_numpy().astype(float))
    df["m_newcust_frac_24h"] = s / (df["m_cnt_24h"] + 1)
    _, s = _window_stats(key, t, HOUR, df["log_amt_ratio"].to_numpy())
    df["m_amt_ratio_sum_1h"] = s

    df = df.sort_values(["device_id_code", "t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    d_gap = df["d_dt_prev"]
    df["d_gap_lt_1h"] = (d_gap < HOUR).astype(np.int8)
    prof = _entity_profile(df, "device_id_code", "d", {
        "gap_lt_1h": df["d_gap_lt_1h"], "log_gap": np.log1p(d_gap.fillna(30 * DAY)),
        "night": df["is_night"], "amt_ratio": df["log_amt_ratio"], "big": df["is_big"],
        "merch_new": df["c_merchant_new"], "loc_new": df["c_location_new"], "n_new": df["c_n_new_entities"],
        "cust_new_on_dev": df["d_cust_new_on_device"], "m_burst": np.log1p(df["m_burst_1h"]),
        "m_small": -np.log1p(df["m_rate"]), "cust_hist_n": np.log1p(df["c_amt_n"]),
        "log_amt": df["log_amt"], "age_dev": df["c_age_dev_abs"].clip(upper=30),
    })
    df = pd.concat([df, prof], axis=1)
    key = df["device_id_code"].to_numpy()
    t = df["t_sec"].to_numpy()
    df["d_cnt_10m"] = _window_stats(key, t, 600, None)

    df = df.sort_values(["customer_id_code", "t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    prof = _entity_profile(df, "customer_id_code", "c", {
        "m_burst": np.log1p(df["m_burst_1h"]), "m_small": -np.log1p(df["m_rate"]),
        "m_gap_lt_1h": df["m_gap_lt_1h"], "dev_shared": df["d_cust_nuniq"].clip(upper=30), "big": df["is_big"],
    })
    df = pd.concat([df, prof], axis=1)
    df["c_m_burst_vs_hist"] = np.log1p(df["m_burst_1h"]) - df["c_hist_m_burst"].fillna(0)
    key = df["customer_id_code"].to_numpy()
    t = df["t_sec"].to_numpy()
    df["c_cnt_10m"] = _window_stats(key, t, 600, None)
    df["m_small"] = -np.log1p(df["m_rate"])

    # ---------------- Global context ----------------
    df = df.sort_values(["t_sec", "_ord"], kind="mergesort").reset_index(drop=True)
    key = np.zeros(len(df), dtype=np.int64)
    t = df["t_sec"].to_numpy()
    df["g_cnt_1h"] = _window_stats(key, t, HOUR, None)
    df["g_cnt_24h"] = _window_stats(key, t, DAY, None)
    df["g_burst_1h"] = df["g_cnt_1h"] / (df["g_cnt_24h"] / 24 + 1)
    _, s = _window_stats(key, t, HOUR, (df["amount_bdt"] > 5000).astype(float).to_numpy())
    df["g_big_share_1h"] = s / (df["g_cnt_1h"] + 1)

    df["night_x_lamt"] = df["is_night"] * df["log_amt"]
    df["night_x_newdev"] = df["is_night"] * df["c_device_new"]
    df["lamt_x_newdev"] = df["log_amt"] * df["c_device_new"]
    df["acct_new"] = (df["account_age_days"] <= 7).astype(np.int8)
    df["log_acct_age"] = np.log1p(df["account_age_days"].clip(lower=0))

    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df.sort_values("_ord").reset_index(drop=True)


def feature_columns(df):
    drop = {"transaction_id", "customer_id", "merchant_id", "device_id", "timestamp", "fraud", "is_test",
            "_ord", "t_sec", "customer_id_code", "merchant_id_code", "device_id_code", "location_code",
            "c_age_expected"}
    return [c for c in df.columns if c not in drop]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
REGIME_START = "2026-06-29"   # the week the model's accuracy collapses in every fold
TEST_SPAN_DAYS = 61
TE_LAGS = [15, 30, 45]            # both are built; experiments choose which to use
TE_WINDOWS = [None, 7, 30]       # None = all history before the lag; 30 = a 30-day window
TE_ALPHA = 20.0               # Bayesian smoothing strength toward the prior
INNER_VALID_DAYS = 45
SWEEP_SEEDS = [0, 1]
FINAL_SEEDS = [0, 1, 2, 3, 4]


def gpu_available():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


DEVICE = "cuda" if gpu_available() else "cpu"
print(f"training device: {DEVICE}")

XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist", device=DEVICE,
    learning_rate=0.018, max_depth=10, min_child_weight=8, subsample=0.90, colsample_bytree=0.80,
    reg_lambda=2.0, reg_alpha=0.15, max_bin=256, max_cat_to_onehot=1,
)
LGBM_PARAMS = dict(
    objective="binary", metric="average_precision", learning_rate=0.018,
    num_leaves=255, min_child_samples=20, subsample=0.90, subsample_freq=1,
    colsample_bytree=0.80, reg_lambda=2.0, reg_alpha=0.10, max_bin=255, force_col_wise=True,
    verbosity=-1, n_jobs=-1,
)
CAT_PARAMS = dict(
    loss_function="Logloss", learning_rate=0.025, depth=10, l2_leaf_reg=4.0,
    random_strength=1.0, border_count=254,
    eval_metric="AUC" if DEVICE == "cuda" else "PRAUC",
    task_type="GPU" if DEVICE == "cuda" else "CPU", verbose=False,
)


# --------------------------------------------------------------------------------------
# LAGGED TARGET ENCODING - the aggressive bet.
#
# Historical fraud rate per merchant / device / customer / location is normally the single
# strongest feature in fraud detection, and the stable model uses none of it. The cold-start
# report is why it is worth the risk here: in the 61-day holdout, transactions on a device
# never seen before were 96% fraud, and on an unseen merchant 34% fraud, against a 1.6% base
# rate. Entity identity carries enormous signal in this dataset.
#
# THE TRAP, and how it is handled.
# A naive expanding fraud rate is fresh for training rows and frozen for test rows: a
# training row at 2026-03-01 sees labels up to yesterday, while a test row at 2026-09-01
# sees nothing after 2026-07-15. The model learns to trust a feature that is 45 days staler
# at prediction time than at training time, and the encoding degrades exactly where it is
# needed. So EVERY row, train and test alike, is given the same handicap: row at time t uses
# only labels from rows strictly before t - LAG.
#
#   train row  2026-03-01, lag 45  ->  labels before 2026-01-15
#   test  row  2026-07-20, lag 45  ->  labels before 2026-06-05
#   test  row  2026-09-15, lag 45  ->  labels before 2026-08-01, but none exist past
#                                       2026-07-15, so effectively 62 days of lag
#
# With lag 45 the effective test lag runs 45 -> 62 days across the test period against a
# flat 45 in training. Close enough that the feature means roughly the same thing on both
# sides, which is the whole point.
#
# The count of labelled rows behind each encoding is also exposed, so the model can learn
# how much to trust a rate built on three observations versus three hundred.
#
# Test rows contribute 0 to both the count and the sum because their label is NaN - so no
# label crosses from test to train, and in the harness the holdout is marked as test and is
# equally invisible.
# --------------------------------------------------------------------------------------
def _lagged_sums(key, t, values, lag, window=None, span_buf_days=400):
    """Sum of `values` over rows j with the same key and t_j < t_i - lag
    (and t_j >= t_i - lag - window when a window is given).
    `key`, `t`, `values` must already be sorted by (key, t)."""
    span = np.int64(t.max() - t.min() + span_buf_days * DAY)
    comb = key * span + (t - t.min())
    hi = np.searchsorted(comb, comb - lag, side="left")
    if window is None:
        lo = np.searchsorted(comb, key * span, side="left")   # start of this key's block
    else:
        lo = np.searchsorted(comb, comb - lag - window, side="left")
    lo = np.minimum(lo, hi)
    cs = np.concatenate([[0.0], np.cumsum(values)])
    return cs[hi] - cs[lo]


TE_ENTITIES = [("merchant_id_code", "m"), ("device_id_code", "d"),
               ("customer_id_code", "c"), ("location_code", "l")]


def add_target_encoding(df, lags=TE_LAGS, windows=TE_WINDOWS, alpha=TE_ALPHA, verbose=True):
    t0 = time.time()
    t = df["t_sec"].to_numpy()
    ordv = df["_ord"].to_numpy()
    y = df["fraud"].fillna(0.0).to_numpy(float)          # test/holdout rows contribute 0
    has = df["fraud"].notna().to_numpy().astype(float)   # ...and are not counted either
    inv = np.empty(len(df), dtype=np.int64)
    made = []

    for lag_d in lags:
        lag = lag_d * DAY
        zero = np.zeros(len(df), np.int64)
        o = np.lexsort((ordv, t, zero))
        inv[o] = np.arange(len(o))
        gn = _lagged_sums(zero[o], t[o], has[o], lag)[inv]
        gs = _lagged_sums(zero[o], t[o], y[o], lag)[inv]
        prior = (gs + 1.0) / (gn + 100.0)   # expanding, equally lagged global fraud rate

        for ent, pfx in TE_ENTITIES:
            key = df[ent].to_numpy()
            o = np.lexsort((ordv, t, key))
            inv[o] = np.arange(len(o))
            for w in windows:
                ws = None if w is None else w * DAY
                n = _lagged_sums(key[o], t[o], has[o], lag, ws)[inv]
                s = _lagged_sums(key[o], t[o], y[o], lag, ws)[inv]
                tag = "all" if w is None else f"{w}d"
                a, b = f"{pfx}_te{lag_d}_rate_{tag}", f"{pfx}_te{lag_d}_n_{tag}"
                df[a] = (s + prior * alpha) / (n + alpha)
                df[b] = np.log1p(n)
                made += [a, b]
    if verbose:
        print(f"  target encoding: {len(made)} columns, lags {lags}, "
              f"windows {windows} ({time.time() - t0:.0f}s)")
    return df, made


def te_columns(cols, lag):
    return [c for c in cols if f"_te{lag}_" in c]


def all_te_columns(cols):
    return [c for c in cols if "_te" in c and "_rate_" in c or "_te" in c and "_n_" in c]


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def fit_xgb(Xes, yes, wes, Xev, yev, Xfit, yfit, wfit, Xho, params, seeds, max_rounds=5000):
    p = dict(XGB_PARAMS); p.update(params)
    des = xgb.DMatrix(Xes, yes, weight=wes, enable_categorical=True)
    dev = xgb.DMatrix(Xev, yev, enable_categorical=True)
    m = xgb.train(dict(p, seed=0), des, max_rounds, evals=[(dev, "v")],
                  early_stopping_rounds=200, verbose_eval=False)
    best = m.best_iteration
    ap = average_precision_score(yev, m.predict(dev, iteration_range=(0, best + 1)))
    del des, dev
    dfit = xgb.DMatrix(Xfit, yfit, weight=wfit, enable_categorical=True)
    dho = xgb.DMatrix(Xho, enable_categorical=True)
    pred = np.zeros(len(Xho))
    for s in seeds:
        pred += xgb.train(dict(p, seed=s), dfit, int(best * 1.15)).predict(dho) / len(seeds)
    return pred, best, ap


def fit_lgbm(Xes, yes, wes, Xev, yev, Xfit, yfit, wfit, Xho, params, seeds, max_rounds=5000):
    import lightgbm as lgb
    p = dict(LGBM_PARAMS); p.update(params)
    dtr = lgb.Dataset(Xes, yes, weight=wes)
    m = lgb.train(p, dtr, max_rounds, valid_sets=[lgb.Dataset(Xev, yev, reference=dtr)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    best = m.best_iteration
    ap = average_precision_score(yev, m.predict(Xev, num_iteration=best))
    dfit = lgb.Dataset(Xfit, yfit, weight=wfit, free_raw_data=False)   # built once, not per seed
    pred = np.zeros(len(Xho))
    for s in seeds:
        mm = lgb.train(dict(p, seed=s, bagging_seed=s, feature_fraction_seed=s),
                       dfit, num_boost_round=int(best * 1.15))
        pred += mm.predict(Xho) / len(seeds)
    return pred, best, ap


def fit_cat(Xes, yes, wes, Xev, yev, Xfit, yfit, wfit, Xho, params, seeds, max_rounds=5000):
    from catboost import CatBoostClassifier, Pool
    cats = [c for c in Xes.columns if str(Xes[c].dtype) == "category"]

    def prep(X):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(str).fillna("__NA__")
        return X

    Xes, Xev, Xfit, Xho = prep(Xes), prep(Xev), prep(Xfit), prep(Xho)
    p = dict(CAT_PARAMS); p.update(params)
    m = CatBoostClassifier(iterations=max_rounds, early_stopping_rounds=200, **p)
    m.fit(Pool(Xes, yes, cat_features=cats, weight=wes), eval_set=Pool(Xev, yev, cat_features=cats))
    best = max(m.get_best_iteration(), 50)
    ap = average_precision_score(yev, m.predict_proba(Xev)[:, 1])
    pool = Pool(Xfit, yfit, cat_features=cats, weight=wfit)
    pred = np.zeros(len(Xho))
    for s in seeds:
        mm = CatBoostClassifier(iterations=int(best * 1.15), random_seed=s, **p)
        mm.fit(pool)
        pred += mm.predict_proba(Xho)[:, 1] / len(seeds)
    return pred, best, ap


FITTERS = {"xgb": fit_xgb, "lgbm": fit_lgbm, "cat": fit_cat}


def blend_and_calibrate(preds, weights=None):
    from scipy.stats import rankdata
    if len(preds) == 1:
        return preds[0]
    r = np.array([rankdata(p) / len(p) for p in preds])
    blend = r.mean(axis=0) if weights is None else (r * (np.asarray(weights, float) /
                                                    np.sum(weights))[:, None]).sum(axis=0)
    reference = np.mean(preds, axis=0)
    return np.sort(reference)[np.argsort(np.argsort(blend))]


def recency_weights(ts, half_life_days):
    if half_life_days is None:
        return None
    age = (ts.max() - ts).dt.total_seconds().to_numpy() / DAY
    return (0.5 ** (age / float(half_life_days))).astype(np.float32)


def paired_delta(y, p_base, p_new, n_boot=300, seed=1):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = (average_precision_score(y[idx], p_new[idx]) -
                average_precision_score(y[idx], p_base[idx])) if y[idx].sum() else np.nan
    return float(np.nanmean(d)), float(np.nanstd(d)), float(np.nanmean(d > 0))


# --------------------------------------------------------------------------------------
# Folds.
#
# REGIME fold is the primary one and is the reason this file exists. Every earlier rejection
# - recency weighting most of all - was measured on a 61-day holdout of which only 17 days
# (28%) were in the post-2026-06-29 regime. A change that helps the new regime and hurts the
# old one gets scored as a loss there, even though the old regime does not exist in test.csv.
# This fold trains on everything before 2026-06-29 and scores ONLY the new regime, which is
# the closest thing in train.csv to what test.csv actually is.
#
# GUARD fold is the familiar 61-day window. Its job is to catch a config that wins the regime
# fold by wrecking everything else. A candidate must not collapse here.
# --------------------------------------------------------------------------------------
_SPLITS = {}


def prepare_fold(full, fold, span_buf, verbose=True):
    ckey = (fold, span_buf)
    if ckey in _SPLITS:
        return _SPLITS[ckey]
    global SPAN_BUFFER_DAYS
    SPAN_BUFFER_DAYS = span_buf

    end = full["timestamp"].max()
    start = pd.Timestamp(REGIME_START) if fold == "regime" else end - pd.Timedelta(days=TEST_SPAN_DAYS)
    tr_raw = full[full["timestamp"] < start].reset_index(drop=True)
    ho_raw = full[full["timestamp"] >= start].reset_index(drop=True)
    y_map = ho_raw.set_index("transaction_id")["fraud"].astype(int)

    t0 = time.time()
    df = build_features(tr_raw, ho_raw)
    df, te_cols = add_target_encoding(df, verbose=verbose)
    base_cols = feature_columns(df)
    assert df.loc[df.is_test == 1, "fraud"].isna().all(), "holdout labels leaked into FE"

    tr = df[df.is_test == 0].reset_index(drop=True)
    ho = df[df.is_test == 1].reset_index(drop=True)
    ts_tr = pd.to_datetime(tr.timestamp)
    if verbose:
        print(f"  {fold:<7} train {tr_raw.timestamp.min().date()} -> {tr_raw.timestamp.max().date()} "
              f"n={len(tr_raw):,} | holdout {ho_raw.timestamp.min().date()} -> "
              f"{ho_raw.timestamp.max().date()} n={len(ho_raw):,} pos={int(y_map.sum()):,} "
              f"| {len(base_cols)} cols, span_buf={span_buf}d ({time.time() - t0:.0f}s)")

    sp = dict(tr=tr, ho=ho, y=tr.fraud.astype(int).to_numpy(),
              y_ho=y_map.loc[ho.transaction_id.values].to_numpy(),
              ts_tr=ts_tr, base_cols=base_cols, te_cols=te_cols,
              inner_mask=(ts_tr < ts_tr.max() - pd.Timedelta(days=INNER_VALID_DAYS)).to_numpy())
    _SPLITS[ckey] = sp
    return sp


def run(cfg, sp):
    t0 = time.time()
    cols = [c for c in sp["base_cols"] if c not in sp["te_cols"]]
    if cfg.get("te_lag"):
        cols += te_columns(sp["te_cols"], cfg["te_lag"])

    tr, y, im = sp["tr"], sp["y"], sp["inner_mask"]
    keep = np.ones(len(tr), bool)
    if cfg.get("train_window_days"):
        keep = (sp["ts_tr"] >= sp["ts_tr"].max() - pd.Timedelta(days=cfg["train_window_days"])).to_numpy()
    w = recency_weights(sp["ts_tr"], cfg.get("half_life"))

    fit_m, es_m = keep, keep & im
    Xes, yes = tr.loc[es_m, cols], y[es_m]
    Xev, yev = tr.loc[keep & ~im, cols], y[keep & ~im]
    Xfit, yfit = tr.loc[fit_m, cols], y[fit_m]
    wes = None if w is None else w[es_m]
    wfit = None if w is None else w[fit_m]

    preds = []
    for name in cfg.get("models", ["xgb"]):
        try:
            p, best, ap = FITTERS[name](Xes, yes, wes, Xev, yev, Xfit, yfit, wfit,
                                        sp["ho"][cols], cfg.get(f"{name}_params", {}), cfg["seeds"])
        except ImportError:
            print(f"      {name} skipped (not installed)")
            continue
        preds.append(p)
        print(f"      {name:<5} best_iter={best:<5} inner_AP={ap:.5f} ({time.time() - t0:.0f}s)")
    if not preds:
        return None
    return blend_and_calibrate(preds), len(cols)


# --------------------------------------------------------------------------------------
# Sweep. XGB-only and 2 seeds: fast, GPU, and the RELATIVE ordering of feature and parameter
# changes is what matters here. The blend is applied once, at submission time.
# --------------------------------------------------------------------------------------
EXPERIMENTS = [
    dict(name="00_base",          te_lag=None, span_buf=10),
    dict(name="01_spanfix",       te_lag=None, span_buf=100),
    dict(name="02_te45",          te_lag=45,   span_buf=100),
    dict(name="03_te30",          te_lag=30,   span_buf=100),
    dict(name="04_te45_rec21",    te_lag=45,   span_buf=100, half_life=21),
    dict(name="05_te45_last90",   te_lag=45,   span_buf=100, train_window_days=90),
    dict(name="06_te45_depth10",  te_lag=45,   span_buf=100, xgb_params=dict(max_depth=10)),
    dict(name="07_te45_depth12",  te_lag=45,   span_buf=100,
         xgb_params=dict(max_depth=12, min_child_weight=40)),
    dict(name="08_te45_lam20",    te_lag=45,   span_buf=100,
         xgb_params=dict(reg_lambda=20.0, colsample_bytree=0.5)),
    dict(name="09_rec21_note",    te_lag=None, span_buf=100, half_life=21),
]
DEFAULTS = dict(te_lag=45, span_buf=100, half_life=None, train_window_days=None,
                models=["xgb"], seeds=SWEEP_SEEDS)


def main_sweep():
    full = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    rows, base = [], {}

    for fold in ["regime", "guard"]:
        print("\n" + "#" * 88 + f"\n# FOLD: {fold}\n" + "#" * 88)
        for raw in EXPERIMENTS:
            cfg = dict(DEFAULTS); cfg.update(raw)
            sp = prepare_fold(full, fold, cfg["span_buf"])
            print(f"\n  >>> [{fold}] {cfg['name']}")
            out = run(cfg, sp)
            if out is None:
                continue
            pred, ncols = out
            ap = average_precision_score(sp["y_ho"], pred)
            if cfg["name"] == EXPERIMENTS[0]["name"]:
                base[fold] = pred
            dm, ds, pw = paired_delta(sp["y_ho"], base[fold], pred)
            v = "KEEP" if pw >= 0.90 else ("borderline" if pw >= 0.75 else "reject")
            print(f"      AP={ap:.5f}  delta={dm:+.5f} (+-{ds:.5f})  P(better)={pw:.2f} -> {v}")
            rows.append(dict(fold=fold, run=cfg["name"], ap=ap, delta=dm, p_better=pw,
                             n_features=ncols))

    b = pd.DataFrame(rows)
    b.to_csv(os.path.join(OUT_DIR, "aggressive_log.csv"), index=False)
    r = b[b.fold == "regime"].set_index("run")
    g = b[b.fold == "guard"].set_index("run")
    print("\n" + "=" * 88)
    print("AGGRESSIVE SWEEP - ranked by the REGIME fold, which is the fold that resembles test.csv")
    print("=" * 88)
    print(f"{'run':<18}{'regime AP':>11}{'delta':>10}{'P':>7}   {'guard AP':>10}{'delta':>10}{'P':>7}")
    print("-" * 88)
    for run_ in r.sort_values("ap", ascending=False).index:
        rr, gg = r.loc[run_], g.loc[run_] if run_ in g.index else None
        gs = (f"{gg.ap:>10.5f}{gg.delta:>+10.5f}{gg.p_better:>7.2f}" if gg is not None
              else f"{'-':>10}{'-':>10}{'-':>7}")
        print(f"{run_:<18}{rr.ap:>11.5f}{rr.delta:>+10.5f}{rr.p_better:>7.2f}   {gs}")
    print("=" * 88)
    print("SHIP a config only if: regime P >= 0.90, AND guard delta is not a collapse")
    print("(a small guard loss is acceptable - the old regime is not in test.csv - but a")
    print("guard delta below about -0.010 means the config broke the model generally).")
    print("The regime fold has ~1,000 positives, so its SE is near 0.02. Only large effects")
    print("are visible; a +0.002 result here is not distinguishable from nothing.")


# --------------------------------------------------------------------------------------
# ===== AGGRESSIVE SUBMISSION =====
#     python aggressive.py submit
# Set WINNER from the sweep before running this.
# --------------------------------------------------------------------------------------
# Extreme probe: deliberately different from the shipped 0.55941 model.
# 5 seeds is intentional: your earlier tests showed that going beyond 5 can backfire.
WINNER = dict(te_lag=45, span_buf=100, half_life=None, train_window_days=None,
              models=["xgb", "lgbm", "cat"], blend_weights=[1, 2, 1], seeds=FINAL_SEEDS,
              xgb_params=dict(max_depth=10, min_child_weight=8, subsample=0.90,
                              colsample_bytree=0.80, reg_lambda=2.0, reg_alpha=0.15),
              lgbm_params=dict(num_leaves=255, min_child_samples=20, subsample=0.90,
                               colsample_bytree=0.80, reg_lambda=2.0, reg_alpha=0.10),
              cat_params=dict(depth=10, l2_leaf_reg=4.0, random_strength=1.0))


def main_submit():
    t0 = time.time()
    global SPAN_BUFFER_DAYS
    SPAN_BUFFER_DAYS = WINNER["span_buf"]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # Competition-safety assertions: test has no usable fraud labels and is strictly later.
    train_ts = pd.to_datetime(train["timestamp"])
    test_ts = pd.to_datetime(test["timestamp"])
    assert test_ts.min() > train_ts.max(), (
        "Expected a strictly chronological train/test split; refusing to run if it is not."
    )
    assert "fraud" not in test.columns or test["fraud"].isna().all(), (
        "Test fraud labels must be absent/NaN."
    )

    df = build_features(train, test)
    # TE is historical-label-only: test rows have NaN fraud and contribute neither count nor sum.
    df, te_cols = add_target_encoding(df)
    cols = [c for c in feature_columns(df) if c not in te_cols]
    # EXTREME: unlike the shipped model, use the full short/mid/long TE bank.
    # This is intentionally high variance: 15d + 30d + 45d lagged rates/counts,
    # with 7d/30d/all-history windows. No test label participates.
    cols += te_cols
    tr = df[df.is_test == 0].reset_index(drop=True)
    te = df[df.is_test == 1].reset_index(drop=True)
    y = tr.fraud.astype(int).to_numpy()
    ts = pd.to_datetime(tr.timestamp)
    assert te.fraud.isna().all(), "test labels leaked into feature building"
    print(f"{len(cols)} features (train {len(tr):,} | test {len(te):,}) in {time.time() - t0:.0f}s")

    keep = np.ones(len(tr), bool)
    if WINNER.get("train_window_days"):
        keep = (ts >= ts.max() - pd.Timedelta(days=WINNER["train_window_days"])).to_numpy()
    im = (ts < ts.max() - pd.Timedelta(days=INNER_VALID_DAYS)).to_numpy()
    w = recency_weights(ts, WINNER.get("half_life"))
    es_m, fit_m = keep & im, keep

    preds = []
    for name in WINNER["models"]:
        try:
            p, best, ap = FITTERS[name](
                tr.loc[es_m, cols], y[es_m], None if w is None else w[es_m],
                tr.loc[keep & ~im, cols], y[keep & ~im],
                tr.loc[fit_m, cols], y[fit_m], None if w is None else w[fit_m],
                te[cols], WINNER.get(f"{name}_params", {}), WINNER["seeds"])
        except ImportError:
            print(f"  {name:<5} SKIPPED (not installed)")
            continue
        preds.append(p)
        print(f"  {name:<5} best_iter={best:<5} valid PR-AUC={ap:.5f} x{len(WINNER['seeds'])} "
              f"seeds ({time.time() - t0:.0f}s)")

    wts = WINNER.get("blend_weights")
    if wts and len(wts) != len(preds):
        wts = None
    final = np.clip(blend_and_calibrate(preds, wts), 0, 1)

    sub = pd.DataFrame({"transaction_id": te.transaction_id.values, "fraud": final})
    sub = sub.set_index("transaction_id").loc[sample.transaction_id].reset_index()
    ok = (len(sub) == len(sample)
          and sub.transaction_id.tolist() == sample.transaction_id.tolist()
          and sub.fraud.notna().all() and sub.fraud.between(0, 1).all()
          and sub.columns.tolist() == ["transaction_id", "fraud"])
    print(f"checks: rows={len(sub):,} ids_match={sub.transaction_id.tolist() == sample.transaction_id.tolist()} "
          f"nan={int(sub.fraud.isna().sum())} range=[{sub.fraud.min():.5f}, {sub.fraud.max():.5f}] "
          f"mean={sub.fraud.mean():.4f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("submission failed validation - not written")
    out = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out, index=False)
    print(f"wrote {out}: {len(sub)} rows from {len(preds)} models, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main_submit()
