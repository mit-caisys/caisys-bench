# Headers for result
QUESTION_HEADER = "Question"
LLM_ANSWER_HEADER = "LLM Answer"
CORRECT_ANSWER_HEADER = "Correct Answer"
LATENCY_HEADER = "Latency"
TOTAL_INPUT_TOKENS_HEADER = "Total Input Tokens"
TOTAL_OUTPUT_TOKENS_HEADER = "Total Output Tokens"

RESULT_COLUMNS_NAMES = [
    QUESTION_HEADER,
    LLM_ANSWER_HEADER,
    CORRECT_ANSWER_HEADER,
    LATENCY_HEADER,
    TOTAL_INPUT_TOKENS_HEADER,
    TOTAL_OUTPUT_TOKENS_HEADER,
]

# Headers for checker
CHECK_ANSWER_HEADER = "Correct"

# Headers for metric
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