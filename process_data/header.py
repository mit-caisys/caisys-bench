POWER_P50 = "POWER_P50"
POWER_P90 = "POWER_P90"
POWER_P99 = "POWER_P99"
ENERGY = "ENERGY"

COLS_ = {
    "GPUTL": {"ymax": 110, "label": "GPU Util. (%)", "op": "mean"},
    "DRAMA": {"ymax": None, "label": "DRAM Active", "op": "mean"},
    "SMACT": {"ymax": None, "label": "SM Active", "op": "mean"},
    "SMOCC": {"ymax": None, "label": "SM Occupancy", "op": "mean"},
    "POWER": {"ymax": None, "label": "Total GPU Power (W)", "op": "sum"},
    "NVLTX": {"ymax": None, "label": "NVLINK (NVLTX)", "op": "sum"},
    "NVLRX": {"ymax": None, "label": "NVLINK (NVLRX)", "op": "sum"},
    "TOTAL_ENERGY_kWh": {
        "ymax": None,
        "label": "GPU Energy (kWh)",
        "op": None,
    },
}
