"""Performance monitoring history (hourly, 14 days) and the alarm log for the
1550 nm service, derived from the hidden case state."""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from .sim import (BER_ALARM, NOW, PM_DAYS, RX_HIGH_ALARM, RX_HIGH_WARN, RX_LOS,
                  RX_LOW_ALARM, RX_LOW_WARN, RX_OVERLOAD, RX_SENSITIVITY, Case, link_loss)

LOS_DBM = -40.0
METRICS = ("rx_power_dbm", "tx_power_dbm", "tx_bias_ma", "pre_fec_ber", "rx_module_temp_c")


def _ber(rx: np.ndarray) -> np.ndarray:
    lg = np.full(rx.shape, -15.0)
    low = rx < RX_SENSITIVITY + 3.0
    lg[low] = -12.0 + 1.6 * (RX_SENSITIVITY - rx[low])
    high = rx > RX_OVERLOAD - 1.5
    lg[high] = np.maximum(lg[high], -12.0 + 4.0 * (rx[high] - RX_OVERLOAD + 1.5) * 1.5)
    lg = np.clip(lg, -15.0, -2.0)
    lg[rx <= LOS_DBM + 1e-9] = -2.0
    return 10.0 ** lg


def history(case: Case) -> dict:
    rng = np.random.default_rng([case.seed, 4242])
    n = PM_DAYS * 24
    times = [NOW - timedelta(hours=n - k) for k in range(n)]
    t_days = np.array([(t - NOW).total_seconds() / 86400.0 for t in times])  # negative
    onset_days = (case.onset - NOW).total_seconds() / 86400.0
    after = t_days >= onset_days
    diurnal = np.sin(2 * np.pi * (t_days * 1.0 + 0.3))
    temp = 36.0 + 3.0 * diurnal + rng.normal(0, 0.3, n)

    ft = case.fault["fault_type"]
    tx = np.full(n, case.tx_dbm)
    bias = case.tx_bias0 + 0.25 * (temp - 36.0) + rng.normal(0, 0.25, n)
    if ft == "tx_degradation":
        eol = case.tx_eol
        frac = np.clip((t_days - onset_days) / eol["days_rise"], 0.0, 1.0)
        bias = bias + eol["bias_rise"] * frac ** 1.5
        t_sat = onset_days + eol["days_rise"]
        ffrac = np.clip((t_days - t_sat) / eol["days_fall"], 0.0, 1.0)
        tx = tx - eol["drop"] * ffrac ** 1.3

    healthy_loss = link_loss(case, 1550, include_fault=False)
    extra = case.fault.get("extra_1550", 0.0)
    if ft in ("dirty_connector", "damaged_connector", "bad_splice"):
        healthy_loss -= extra   # element losses already include the fault
    loss = np.full(n, healthy_loss)
    if ft in ("dirty_connector", "damaged_connector", "bad_splice"):
        loss = loss + extra * after
    elif ft == "macrobend":
        factor = np.array([case.bend_factor(t) for t in times])
        loss = loss + extra * after * factor
    att = np.full(n, case.att_design_db)
    if ft == "wrong_attenuator":
        att = np.where(after, case.att_installed_db, case.att_design_db)

    rx = tx - loss - att - 0.1 + rng.normal(0, 0.06, n)
    if ft == "fibre_break":
        rx = np.where(after, LOS_DBM, rx)
    tx_rep = tx + 0.2 + rng.normal(0, 0.05, n)
    ber = _ber(rx) * 10 ** rng.normal(0, 0.08, n)
    return {"time": times, "rx_power_dbm": rx, "tx_power_dbm": tx_rep, "tx_bias_ma": bias,
            "pre_fec_ber": np.clip(ber, 1e-15, 0.5), "rx_module_temp_c": temp}


def alarms(case: Case, hist: dict = None) -> list:
    hist = hist or history(case)
    rx, ber, times = hist["rx_power_dbm"], hist["pre_fec_ber"], hist["time"]
    rules = [
        ("RX_LOS", "critical", rx <= RX_LOS),
        ("RX_LOW_ALARM", "major", (rx < RX_LOW_ALARM) & (rx > RX_LOS)),
        ("RX_LOW_WARNING", "minor", (rx < RX_LOW_WARN) & (rx >= RX_LOW_ALARM)),
        ("RX_HIGH_ALARM", "major", rx > RX_HIGH_ALARM),
        ("RX_HIGH_WARNING", "minor", (rx > RX_HIGH_WARN) & (rx <= RX_HIGH_ALARM)),
        ("PRE_FEC_BER_HIGH", "major", ber > BER_ALARM),
    ]
    out = []
    for name, sev, cond in rules:
        active, start = False, None
        for k, c in enumerate(cond):
            if c and not active:
                active, start = True, times[k]
            elif not c and active:
                # Require two clear samples to clear (hysteresis).
                if k + 1 < len(cond) and cond[k + 1]:
                    continue
                out.append({"alarm": name, "severity": sev, "raised": start, "cleared": times[k]})
                active = False
        if active:
            out.append({"alarm": name, "severity": sev, "raised": start, "cleared": None})
    out.sort(key=lambda a: a["raised"])
    for a in out:
        a["raised"] = a["raised"].strftime("%Y-%m-%d %H:%M UTC")
        a["cleared"] = a["cleared"].strftime("%Y-%m-%d %H:%M UTC") if a["cleared"] else "active"
        a["entity"] = "B-RX (1550 nm service from site A)"
    return out
