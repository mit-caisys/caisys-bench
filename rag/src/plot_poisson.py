import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.header import (
    CORRECT_HEADER,
    END_TIME_HEADER,
    START_TIME_HEADER,
    TOTAL_INPUT_TOKENS_HEADER,
    TOTAL_OUTPUT_TOKENS_HEADER,
)
from common.paths import POISSON_PLOT_DIR, PROCESSED_METRIC_DIR, RESULT_DIR
from common.prefix import POISSON_PREFIX
from pydantic import BaseModel, ConfigDict, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))

from process_data import ENERGY, POWER_P50, POWER_P90, POWER_P99

QUANTILES = [50, 90, 99]

plt.rcParams.update({"font.size": 12})


class PoissonObject(BaseModel):
    model_config = ConfigDict(frozen=True)
    filename: str
    provider: str = Field(description="LLM provider used to generate result")
    model: str = Field(description="Model used to generate result")
    num_frames_wiki_links: int = Field(
        description="Number of frames Wiki links included to create the vector database. If `runtype == result`, then `k` means questions with indices 0 to `k - 1`"
    )

    rate: float
    run_index: int = Field(description="Unique index for this run")

    # fixed
    chunk_size: int
    chunk_overlap: int

    # independent
    num_docs: int = Field(
        description="Number of retrieved documents from the vector database"
    )

    num_questions: int = Field(description="Number of questions used in the run")
    accuracy: float
    latency_list: list[float] = Field(description="in Seconds")
    total_input_tokens_list: list[int]
    total_output_tokens_list: list[int]
    power_p50: float = Field(description="in Watts")
    power_p90: float = Field(description="in Watts")
    power_p99: float = Field(description="in Watts")
    energy: float = Field(description="in kWh")

    def __str__(self):
        return self.filename

    def __lt__(self, other):
        if self.provider != other.provider:
            return self.provider < other.provider
        if self.model != other.model:
            return self.model < other.model
        if self.num_frames_wiki_links != other.num_frames_wiki_links:
            return self.num_frames_wiki_links < other.num_frames_wiki_links
        if self.rate != other.rate:
            return self.rate < other.rate
        if self.run_index != other.run_index:
            return self.run_index < other.run_index
        if self.chunk_size != other.chunk_size:
            return self.chunk_size < other.chunk_size
        if self.chunk_overlap != other.chunk_overlap:
            return self.chunk_overlap < other.chunk_overlap
        if self.num_docs != other.num_docs:
            return self.num_docs < other.num_docs
        return False


class PoissonMergeObject(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str = Field(description="LLM provider used to generate result")
    model: str = Field(description="Model used to generate result")
    num_frames_wiki_links: int = Field(
        description="Number of frames Wiki links included to create the vector database. If `runtype == result`, then `k` means questions with indices 0 to `k - 1`"
    )

    rate: float

    # fixed
    chunk_size: int
    chunk_overlap: int

    # independent
    num_docs: int = Field(
        description="Number of retrieved documents from the vector database"
    )

    num_questions: int = Field(description="Number of questions used in the run")
    accuracy: list[float]
    latency_list: list[list[float]] = Field(description="in Seconds")
    total_input_tokens_list: list[list[int]]
    total_output_tokens_list: list[list[int]]
    power_p50: list[float] = Field(description="in Watts")
    power_p90: list[float] = Field(description="in Watts")
    power_p99: list[float] = Field(description="in Watts")
    energy_list: list[float] = Field(description="in kWh")

    def __str__(self):
        return f"{POISSON_PREFIX}_{str(self.rate).replace('.', '-')}{self.provider}_{self.model}_{self.chunk_size}_{self.chunk_overlap}_{self.num_docs}_{self.num_frames_wiki_links}"

    def __lt__(self, other):
        if self.provider != other.provider:
            return self.provider < other.provider
        if self.model != other.model:
            return self.model < other.model
        if self.num_frames_wiki_links != other.num_frames_wiki_links:
            return self.num_frames_wiki_links < other.num_frames_wiki_links
        if self.rate != other.rate:
            return self.rate < other.rate
        if self.chunk_size != other.chunk_size:
            return self.chunk_size < other.chunk_size
        if self.chunk_overlap != other.chunk_overlap:
            return self.chunk_overlap < other.chunk_overlap
        if self.num_docs != other.num_docs:
            return self.num_docs < other.num_docs
        return False


class PoissonIdentifier(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str
    model: str
    chunk_size: int
    chunk_overlap: int
    num_docs: int
    num_frames_wiki_links: int

    def __str__(self):
        return f"{self.provider}_{self.model}_{self.chunk_size}_{self.chunk_overlap}_{self.num_docs}_{self.num_frames_wiki_links}"

    def to_legend(self):
        return f"{self.num_docs} Retrieved Chunks"

    def __lt__(self, other):
        if self.provider != other.provider:
            return self.provider < other.provider
        if self.model != other.model:
            return self.model < other.model
        if self.chunk_size != other.chunk_size:
            return self.chunk_size < other.chunk_size
        if self.chunk_overlap != other.chunk_overlap:
            return self.chunk_overlap < other.chunk_overlap
        if self.num_docs != other.num_docs:
            return self.num_docs < other.num_docs
        if self.num_frames_wiki_links != other.num_frames_wiki_links:
            return self.num_frames_wiki_links < other.num_frames_wiki_links
        return False


def create_poisson_object(
    csv_path: Path,
    check_answer_header: str = CORRECT_HEADER,
    start_time_header: str = START_TIME_HEADER,
    end_time_header: str = END_TIME_HEADER,
    total_input_tokens_header: str = TOTAL_INPUT_TOKENS_HEADER,
    total_output_tokens_header: str = TOTAL_OUTPUT_TOKENS_HEADER,
) -> PoissonObject:
    base_name = csv_path.stem
    parts = base_name.split("_")[:-1]  # remove cores information

    df = pd.read_csv(csv_path)
    num_questions = len(df)
    num_correct = df[check_answer_header].sum()

    stats_files = list((PROCESSED_METRIC_DIR / base_name).glob(f"*.json"))
    assert len(stats_files) <= 1, "Too many stats files"
    power_p50 = None
    power_p90 = None
    power_p99 = None
    energy = None
    if len(stats_files) > 0:
        with open(stats_files[0], "r") as stats_file:
            stats = json.load(stats_file)
            power_p50 = stats[POWER_P50]
            power_p90 = stats[POWER_P90]
            power_p99 = stats[POWER_P99]
            energy = stats[ENERGY]

    return PoissonObject(
        filename=base_name,
        provider=parts[2],
        model=parts[3],
        num_frames_wiki_links=int(parts[9]),
        rate=float(parts[1].replace("-", ".")),
        run_index=int(parts[4]),
        chunk_size=int(parts[6]),
        chunk_overlap=int(parts[7]),
        num_docs=int(parts[8]),
        num_questions=num_questions,
        accuracy=num_correct / num_questions,
        latency_list=(df[end_time_header] - df[start_time_header])
        .sort_values()
        .to_list(),
        total_input_tokens_list=df[total_input_tokens_header].sort_values().to_list(),
        total_output_tokens_list=df[total_output_tokens_header].sort_values().to_list(),
        power_p50=power_p50,
        power_p90=power_p90,
        power_p99=power_p99,
        energy=energy,
    )


def merge_poisson_objects(
    poisson_objects: List[PoissonObject],
) -> List[PoissonMergeObject]:
    merged_dict = defaultdict(list)

    for obj in poisson_objects:
        key = (
            obj.provider,
            obj.model,
            obj.num_frames_wiki_links,
            obj.rate,
            obj.chunk_size,
            obj.chunk_overlap,
            obj.num_docs,
        )
        merged_dict[key].append(obj)

    merged_objects = []
    for key, group in merged_dict.items():
        (
            provider,
            model,
            num_frames_wiki_links,
            rate,
            chunk_size,
            chunk_overlap,
            num_docs,
        ) = key

        merged = PoissonMergeObject(
            provider=provider,
            model=model,
            num_frames_wiki_links=num_frames_wiki_links,
            rate=rate,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_docs=num_docs,
            num_questions=sum(obj.num_questions for obj in group),
            accuracy=[obj.accuracy for obj in group],
            latency_list=[obj.latency_list for obj in group],
            total_input_tokens_list=[obj.total_input_tokens_list for obj in group],
            total_output_tokens_list=[obj.total_output_tokens_list for obj in group],
            power_p50=[obj.power_p50 for obj in group],
            power_p90=[obj.power_p90 for obj in group],
            power_p99=[obj.power_p99 for obj in group],
            energy_list=[obj.energy for obj in group],
        )
        merged_objects.append(merged)

    return merged_objects


def find_and_extract_poissons(
    prefixes: list[str] = [POISSON_PREFIX],
    result_dir: Path = RESULT_DIR,
) -> list[PoissonMergeObject]:
    if not result_dir.is_dir():
        print(f"Error: '{result_dir}' is not a valid directory.")
        return []

    poissons = []
    for file_path in result_dir.iterdir():
        if (
            file_path.is_file()
            and (file_path.name).startswith(tuple(prefixes))
            and file_path.suffix == ".csv"
            and "split" in file_path.stem.split("_")
        ):
            poissons.append(create_poisson_object(file_path))
    return sorted(merge_poisson_objects(poissons))


def filter_poisson_result(
    result: PoissonMergeObject, poisson_identifier: PoissonIdentifier
) -> bool:
    return (
        result.provider == poisson_identifier.provider
        and result.model == poisson_identifier.model
        and result.chunk_size == poisson_identifier.chunk_size
        and result.chunk_overlap == poisson_identifier.chunk_overlap
        and result.num_docs == poisson_identifier.num_docs
        and result.num_frames_wiki_links == poisson_identifier.num_frames_wiki_links
    )


def get_poisson_identifiers(
    results: list[PoissonMergeObject], filter_result=lambda _: True
) -> list[PoissonIdentifier]:
    return sorted(
        list(
            {
                PoissonIdentifier(
                    provider=result.provider,
                    model=result.model,
                    chunk_size=result.chunk_size,
                    chunk_overlap=result.chunk_overlap,
                    num_docs=result.num_docs,
                    num_frames_wiki_links=result.num_frames_wiki_links,
                )
                for result in results
                if filter_result(result)
            }
        )
    )


def get_poissons_list(
    results: list[PoissonMergeObject],
    poisson_identifiers: list[PoissonIdentifier],
    filter_result=lambda x, y: True,
    extra_filter_result=lambda x: True,
):
    return {
        poisson_identifier: [
            result
            for result in results
            if filter_result(result, poisson_identifier) and extra_filter_result(result)
        ]
        for poisson_identifier in poisson_identifiers
    }


def _error_bar_plot(
    x_vals: list[float],
    y_vals: list[float],
    y_err_lowers: list[float],
    y_err_uppers: list[float],
    label: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    plt.figure()
    plt.errorbar(
        x_vals,
        y_vals,
        fmt="o",
        capsize=5,
        yerr=[y_err_lowers, y_err_uppers],
        label=label,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0)
    plt.legend()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def _generate_error_bar_data(
    results: list[PoissonMergeObject],
    x_fn: Callable,
    y_fn: Callable,
    filter_fn: Callable = lambda r: True,
):
    x_vals, y_vals, y_err_lowers, y_err_uppers = (
        [],
        [],
        [],
        [],
    )
    for result in results:
        if not filter_fn(result):
            continue
        x_vals.append(x_fn(result))
        ys = sorted(y_fn(result))
        num_ys = len(ys)
        med = (ys[num_ys // 2] + ys[(num_ys - 1) // 2]) / 2
        y_vals.append(med)
        y_err_lowers.append(med - ys[0])
        y_err_uppers.append(ys[-1] - med)
    return x_vals, y_vals, y_err_lowers, y_err_uppers


def plot_latency_vs_rate(results: list[PoissonMergeObject], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    poisson_identifiers = get_poisson_identifiers(results)
    poissons_list = get_poissons_list(
        results,
        poisson_identifiers,
        filter_poisson_result,
        lambda result: result.rate <= 0.2,
    )
    for quantile in QUANTILES:
        (dest_dir / f"p{quantile}").mkdir(exist_ok=True)
        x_fn = lambda r: r.rate
        y_fn = lambda r: [
            np.percentile(latency_list, quantile) for latency_list in r.latency_list
        ]
        xlabel = "Poisson Rate"
        ylabel = f"Latency P{quantile} in seconds"
        title = f"Latency P{quantile} vs Poisson Rate"

        for poisson_identifier in poisson_identifiers:
            x_vals, y_vals, y_err_lowers, y_err_uppers = _generate_error_bar_data(
                poissons_list[poisson_identifier], x_fn, y_fn
            )
            _error_bar_plot(
                x_vals,
                y_vals,
                y_err_lowers,
                y_err_uppers,
                poisson_identifier.to_legend(),
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                save_path=dest_dir / f"p{quantile}" / f"{str(poisson_identifier)}.pdf",
            )

        plt.figure()
        for poisson_identifier in poisson_identifiers:
            x_vals, y_vals, y_err_lowers, y_err_uppers = _generate_error_bar_data(
                poissons_list[poisson_identifier], x_fn, y_fn
            )

            plt.errorbar(
                x_vals,
                y_vals,
                fmt="o",
                capsize=5,
                yerr=[y_err_lowers, y_err_uppers],
                label=poisson_identifier.to_legend(),
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(0)
        plt.legend()
        plt.savefig(
            dest_dir / f"p{quantile}" / "comparison.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_energy_vs_rate(results: list[PoissonMergeObject], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    poisson_identifiers = get_poisson_identifiers(results)
    poissons_list = get_poissons_list(
        results,
        poisson_identifiers,
        filter_poisson_result,
        lambda result: result.rate <= 0.2,
    )

    x_fn = lambda r: r.rate
    y_fn = lambda r: r.energy_list
    xlabel = "Poisson Rate"
    ylabel = f"Energy in kWh"
    title = f"Energy vs Poisson Rate"

    for poisson_identifier in poisson_identifiers:
        x_vals, y_vals, y_err_lowers, y_err_uppers = _generate_error_bar_data(
            poissons_list[poisson_identifier], x_fn, y_fn
        )
        _error_bar_plot(
            x_vals,
            y_vals,
            y_err_lowers,
            y_err_uppers,
            poisson_identifier.to_legend(),
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            save_path=dest_dir / f"{str(poisson_identifier)}.pdf",
        )

    plt.figure()
    for poisson_identifier in poisson_identifiers:
        x_vals, y_vals, y_err_lowers, y_err_uppers = _generate_error_bar_data(
            poissons_list[poisson_identifier], x_fn, y_fn
        )

        plt.errorbar(
            x_vals,
            y_vals,
            fmt="o",
            capsize=5,
            yerr=[y_err_lowers, y_err_uppers],
            label=poisson_identifier.to_legend(),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0)
    plt.legend()
    plt.savefig(
        dest_dir / "comparison.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    results = find_and_extract_poissons()
    POISSON_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_latency_vs_rate(results, POISSON_PLOT_DIR / "latency_vs_rate")
    plot_energy_vs_rate(results, POISSON_PLOT_DIR / "energy_vs_rate")


if __name__ == "__main__":
    main()
