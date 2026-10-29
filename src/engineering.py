"""Derived traffic features and a fixed, documented input schema."""
import numpy as np
import pandas as pd

CATEGORICAL = ["proto", "service", "state"]
# Explicit allowlist prevents accidental inclusion of labels, IDs, or output columns.
BASE_NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
    "stcpb", "dtcpb", "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_ftp_login", "is_sm_ips_ports",
    "sttl", "dttl", "sht", "dht", "sintpkt", "dintpkt"
]
ENGINEERED = ["total_bytes", "bytes_per_second", "inbound_outbound_ratio",
              "outbound_fraction", "packet_ratio", "sbytes_per_packet", "dbytes_per_packet"]


def safe_divide(numerator, denominator):
    a = np.asarray(numerator, dtype="float64")
    b = np.asarray(denominator, dtype="float64")
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    s, d = out["sbytes"], out["dbytes"]
    sp, dp = out["spkts"], out["dpkts"]
    out["total_bytes"] = s + d
    out["bytes_per_second"] = safe_divide(out["total_bytes"], out["dur"])
    out["inbound_outbound_ratio"] = safe_divide(d, s)
    out["outbound_fraction"] = safe_divide(s, out["total_bytes"])
    out["packet_ratio"] = safe_divide(sp, dp)
    out["sbytes_per_packet"] = safe_divide(s, sp)
    out["dbytes_per_packet"] = safe_divide(d, dp)
    return out.replace([np.inf, -np.inf], 0)


def feature_columns(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [name for name in BASE_NUMERIC + ENGINEERED if name in train.columns]
    return numeric, CATEGORICAL.copy()


def matrix(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    missing = set(numeric + categorical) - set(frame.columns)
    if missing:
        raise ValueError(f"Split is missing training features: {sorted(missing)}")
    data = frame[numeric + categorical].copy()
    for name in numeric:
        data[name] = pd.to_numeric(data[name], errors="raise")
    data[categorical] = data[categorical].astype(str)
    return data
