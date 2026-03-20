"""
vLLM monitoring module.

Context manager that polls the vLLM /metrics endpoint to collect
prefix cache hit rates, KV cache usage, and block lifetime statistics.
"""

import csv
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests


class VLLMMonitor:
    """
    A context manager to monitor vLLM prefix cache metrics in a background thread.

    Logs a CSV of running average and interval hit rates.
    """

    def __init__(
        self, output_dir, interval_sec=1, vllm_url="http://localhost:8000/metrics"
    ):
        self.output_file = Path(output_dir) / "vllm_cache_log.csv"
        self.interval_sec = interval_sec
        self.vllm_url = vllm_url
        self.thread = None
        self.stop_event = threading.Event()

        self.baseline_metrics = None
        self.prev_metrics = None
        self.start_time = 0

    def _get_vllm_metrics(self):
        """Fetches metrics from the vLLM server."""
        try:
            response = requests.get(self.vllm_url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.ConnectionError:
            print(
                f"Error: [VLLMMonitor] Could not connect to vLLM server at {self.vllm_url}",
                file=sys.stderr,
            )
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: [VLLMMonitor] Error fetching metrics: {e}", file=sys.stderr)
            return None

    def _parse_cache_metrics(self, metrics_text) -> Dict | None:
        """
        Parses Prometheus metrics, correctly handling labels {...}.
        Tries 'gpu_prefix_cache...' first, then 'prefix_cache...'.
        """

        # This regex pattern matches:
        # ^(metric_name)   - start of line, metric name
        # (?:{[^}]+})?     - (optional) non-capturing group for labels {...}
        # \s+              - whitespace
        # ([0-9e\+\.]+)    - the number we want to capture

        def find_metric(name, text):
            pattern = re.compile(
                rf"^{name}(?:{{[^}}]+}})?\s+([0-9eE\+\-\.]+)", re.MULTILINE
            )
            return pattern.search(text)

        hits_match = find_metric(
            "vllm:prefix_cache_hits_total", metrics_text
        ) or find_metric("vllm:gpu_prefix_cache_hits_total", metrics_text)

        queries_match = find_metric(
            "vllm:prefix_cache_queries_total", metrics_text
        ) or find_metric("vllm:gpu_prefix_cache_queries_total", metrics_text)

        kv_cache_usage_match = find_metric(
            "vllm:kv_cache_usage_perc", metrics_text
        ) or find_metric("vllm:gpu_cache_usage_perc", metrics_text)

        kv_block_lifetime_count_match = find_metric(
            "vllm:kv_block_lifetime_seconds_count", metrics_text
        )

        kv_block_lifetime_sum_match = find_metric(
            "vllm:kv_block_lifetime_seconds_sum", metrics_text
        )

        kv_block_idle_before_evict_count_match = find_metric(
            "vllm:kv_block_idle_before_evict_seconds_count", metrics_text
        )

        kv_block_idle_before_evict_sum_match = find_metric(
            "vllm:kv_block_idle_before_evict_seconds_sum", metrics_text
        )

        kv_block_reuse_gap_count_match = find_metric(
            "vllm:kv_block_reuse_gap_seconds_count", metrics_text
        )

        kv_block_reuse_gap_sum_match = find_metric(
            "vllm:kv_block_reuse_gap_seconds_sum", metrics_text
        )

        if not (
            hits_match
            and queries_match
            and kv_cache_usage_match
            and kv_block_lifetime_count_match
            and kv_block_lifetime_sum_match
            and kv_block_idle_before_evict_count_match
            and kv_block_idle_before_evict_sum_match
            and kv_block_reuse_gap_count_match
            and kv_block_reuse_gap_sum_match
        ):
            print(
                "Warning: [VLLMMonitor] Could not find vllm:___cache___ metrics.",
                file=sys.stderr,
            )
            print(
                "         Please double-check the /metrics endpoint. Is prefix caching enabled?",
                file=sys.stderr,
            )
            return None

        try:
            hits = float(hits_match.group(1))
            queries = float(queries_match.group(1))
            kv_cache_usage = float(kv_cache_usage_match.group(1))
            kv_block_lifetime_count = float(kv_block_lifetime_count_match.group(1))
            kv_block_lifetime_sum = float(kv_block_lifetime_sum_match.group(1))
            kv_block_idle_before_evict_count = float(
                kv_block_idle_before_evict_count_match.group(1)
            )
            kv_block_idle_before_evict_sum = float(
                kv_block_idle_before_evict_sum_match.group(1)
            )
            kv_block_reuse_gap_count = float(kv_block_reuse_gap_count_match.group(1))
            kv_block_reuse_gap_sum = float(kv_block_reuse_gap_sum_match.group(1))
            return {
                "hits": hits,
                "queries": queries,
                "kv_cache_usage": kv_cache_usage,
                "kv_block_lifetime_count": kv_block_lifetime_count,
                "kv_block_lifetime_sum": kv_block_lifetime_sum,
                "kv_block_idle_before_evict_count": kv_block_idle_before_evict_count,
                "kv_block_idle_before_evict_sum": kv_block_idle_before_evict_sum,
                "kv_block_reuse_gap_count": kv_block_reuse_gap_count,
                "kv_block_reuse_gap_sum": kv_block_reuse_gap_sum,
            }
        except Exception as e:
            print(f"Error: [VLLMMonitor] Error parsing metrics: {e}", file=sys.stderr)
            return {}

    def _monitor_loop(self):
        """The main loop for the monitoring thread."""

        try:
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "elapsed_sec",
                        "running_avg_hit_rate",
                        "interval_hit_rate",
                        "kv_cache_usage_percent",
                        "kv_block_lifetime_avg",
                        "kv_block_idle_before_evict_avg",
                        "kv_block_reuse_gap_avg",
                        "job_total_hits",
                        "job_total_queries",
                        "kv_block_lifetime_count",
                        "kv_block_lifetime_sum",
                        "kv_block_idle_before_evict_count",
                        "kv_block_idle_before_evict_sum",
                        "kv_block_reuse_gap_count",
                        "kv_block_reuse_gap_sum",
                    ]
                )
        except IOError as e:
            print(f"Error: [VLLMMonitor] Could not open log file: {e}", file=sys.stderr)
            return  # Stop thread if we can't write

        while not self.stop_event.is_set():
            metrics_text = self._get_vllm_metrics()
            if not metrics_text:
                self.stop_event.wait(self.interval_sec)
                continue

            current_metrics = self._parse_cache_metrics(metrics_text)
            if not current_metrics:
                self.stop_event.wait(self.interval_sec)
                continue

            if self.baseline_metrics is None:
                self.baseline_metrics = current_metrics
                self.prev_metrics = current_metrics
                self.start_time = time.time()
                print("[VLLMMonitor] Established baseline. Monitoring job...")
                self.stop_event.wait(self.interval_sec)
                continue

            elapsed_sec = time.time() - self.start_time

            job_total_hits = current_metrics["hits"] - self.baseline_metrics["hits"]
            job_total_queries = (
                current_metrics["queries"] - self.baseline_metrics["queries"]
            )

            job_running_avg_rate = 0.0
            if job_total_queries > 0:
                job_running_avg_rate = job_total_hits / job_total_queries

            interval_hits = current_metrics["hits"] - self.prev_metrics["hits"]
            interval_queries = current_metrics["queries"] - self.prev_metrics["queries"]

            interval_rate = 0.0
            if interval_queries > 0:
                interval_rate = interval_hits / interval_queries

            current_kv_cache_usage_perc = current_metrics["kv_cache_usage"] * 100

            kv_block_lifetime_count = (
                current_metrics["kv_block_lifetime_count"]
                - self.baseline_metrics["kv_block_lifetime_count"]
            )
            kv_block_lifetime_sum = (
                current_metrics["kv_block_lifetime_sum"]
                - self.baseline_metrics["kv_block_lifetime_sum"]
            )
            kv_block_lifetime_avg = 0.0
            if kv_block_lifetime_count > 0:
                kv_block_lifetime_avg = kv_block_lifetime_sum / kv_block_lifetime_count
            kv_block_idle_before_evict_count = (
                current_metrics["kv_block_idle_before_evict_count"]
                - self.baseline_metrics["kv_block_idle_before_evict_count"]
            )
            kv_block_idle_before_evict_sum = (
                current_metrics["kv_block_idle_before_evict_sum"]
                - self.baseline_metrics["kv_block_idle_before_evict_sum"]
            )
            kv_block_idle_before_evict_avg = 0.0
            if kv_block_idle_before_evict_count > 0:
                kv_block_idle_before_evict_avg = (
                    kv_block_idle_before_evict_sum / kv_block_idle_before_evict_count
                )
            kv_block_reuse_gap_count = (
                current_metrics["kv_block_reuse_gap_count"]
                - self.baseline_metrics["kv_block_reuse_gap_count"]
            )
            kv_block_reuse_gap_sum = (
                current_metrics["kv_block_reuse_gap_sum"]
                - self.baseline_metrics["kv_block_reuse_gap_sum"]
            )
            kv_block_reuse_gap_avg = 0.0
            if kv_block_reuse_gap_count > 0:
                kv_block_reuse_gap_avg = (
                    kv_block_reuse_gap_sum / kv_block_reuse_gap_count
                )
            try:
                with open(self.output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            datetime.now().isoformat(),
                            round(elapsed_sec, 2),
                            round(job_running_avg_rate, 4),
                            round(interval_rate, 4),
                            round(current_kv_cache_usage_perc, 4),
                            round(kv_block_lifetime_avg, 4),
                            round(kv_block_idle_before_evict_avg, 4),
                            round(kv_block_reuse_gap_avg, 4),
                            job_total_hits,
                            job_total_queries,
                            kv_block_lifetime_count,
                            kv_block_lifetime_sum,
                            kv_block_idle_before_evict_count,
                            kv_block_idle_before_evict_sum,
                            kv_block_reuse_gap_count,
                            kv_block_reuse_gap_sum,
                        ]
                    )
            except IOError as e:
                print(
                    f"Error: [VLLMMonitor] Could not write to log file: {e}",
                    file=sys.stderr,
                )

            self.prev_metrics = current_metrics

            self.stop_event.wait(self.interval_sec)

    def __enter__(self):
        print("Starting vLLM cache monitor thread...")
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("\nStopping vLLM cache monitor thread...")
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        print(f"vLLM monitor stopped. Log saved to {self.output_file}")
