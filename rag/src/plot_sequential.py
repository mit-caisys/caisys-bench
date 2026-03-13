from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt  # pyright: ignore
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from common.header import (
    CORRECT_HEADER,
    END_TIME_HEADER,
    START_TIME_HEADER,
    TOTAL_INPUT_TOKENS_HEADER,
    TOTAL_OUTPUT_TOKENS_HEADER,
)
from common.path import RESULT_DIR, SEQUENTIAL_PLOT_DIR
from common.prefix import BASELINE_PREFIX, RESULT_PREFIX

QUANTILES = [50, 90, 99]

COLORS = [
    "#268fda",
    "#ff9935",
    "#36c036",
    "#ff2f30",
    "#b47be1",
    "#a8685c",
    "#ff94e7",
    "#999999",
    "#e0e026",
    "#1fe3ff",
]

plt.rcParams.update({"font.size": 12})


def darken_hex_color(hex_color, factor=0.9):
    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return "#{:02x}{:02x}{:02x}".format(r, g, b)


class ResultObject(BaseModel):
    model_config = ConfigDict(frozen=True)
    filename: str
    run_type: Literal["baseline", "result"]
    provider: str = Field(description="LLM provider used to generate result")
    model: str = Field(description="Model used to generate result")
    num_frames_wiki_links: int = Field(
        description="Number of frames Wiki links included to create the vector database. If `runtype == result`, then `k` means questions with indices 0 to `k - 1`"
    )

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

    def __str__(self):
        return self.filename

    def to_label(self):
        model = (
            self.model
            if self.model.split("-")[0] == "gpt"
            else self.model[self.model.find("-") + 1 :]
        )
        if self.run_type == "baseline":
            return f"{model} baseline DB"
        return f"{model} {self.num_frames_wiki_links}-link DB"

    def __lt__(self, other):
        if self.run_type != other.run_type:
            return self.run_type < other.run_type
        if self.provider != other.provider:
            return self.provider < other.provider
        if self.model != other.model:
            return self.model < other.model
        if self.num_frames_wiki_links != other.num_frames_wiki_links:
            return self.num_frames_wiki_links < other.num_frames_wiki_links
        if self.chunk_size != other.chunk_size:
            return self.chunk_size < other.chunk_size
        if self.chunk_overlap != other.chunk_overlap:
            return self.chunk_overlap < other.chunk_overlap
        if self.num_docs != other.num_docs:
            return self.num_docs < other.num_docs
        return False


class Identifier(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_type: Literal["baseline", "result"]
    provider: str
    model: str
    num_frames_wiki_links: int

    def __str__(self):
        model = (
            self.model
            if self.model.split("-")[0] == "gpt"
            else self.model[self.model.find("-") + 1 :]
        )
        if self.run_type == "baseline":
            return f"{model} baseline DB"
        return f"{model} {self.num_frames_wiki_links}-link DB"

    def __lt__(self, other):
        if self.run_type != other.run_type:
            return self.run_type < other.run_type
        if self.provider != other.provider:
            return self.provider < other.provider
        if self.model != other.model:
            return self.model < other.model
        if self.num_frames_wiki_links != other.num_frames_wiki_links:
            return self.num_frames_wiki_links < other.num_frames_wiki_links
        return False


class ModelIdentifier(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str
    model: str
    chunk_size: int
    chunk_overlap: int

    def __str__(self):
        return f"{self.provider}_{self.model}_{self.chunk_size}_{self.chunk_overlap}"

    def __lt__(self, other):
        if self.provider != other.provider:
            return self.provider < other.provider
        if self.model != other.model:
            return self.model < other.model
        if self.chunk_size != other.chunk_size:
            return self.chunk_size < other.chunk_size
        if self.chunk_overlap != other.chunk_overlap:
            return self.chunk_overlap < other.chunk_overlap
        return False


def create_result_object(
    csv_path: Path,
    correct_header: str = CORRECT_HEADER,
    start_time_header: str = START_TIME_HEADER,
    end_time_header: str = END_TIME_HEADER,
    total_input_tokens_header: str = TOTAL_INPUT_TOKENS_HEADER,
    total_output_tokens_header: str = TOTAL_OUTPUT_TOKENS_HEADER,
) -> ResultObject:
    base_name = csv_path.stem
    parts = base_name.split("_")

    df = pd.read_csv(csv_path)
    num_questions = len(df)
    num_correct = (df[correct_header]).sum()

    return ResultObject(
        filename=base_name,
        run_type=parts[0],  # pyright: ignore
        provider=parts[1],
        model=parts[2],
        num_frames_wiki_links=int(parts[7]),
        chunk_size=int(parts[4]),
        chunk_overlap=int(parts[5]),
        num_docs=int(parts[6]),
        num_questions=num_questions,
        accuracy=num_correct / num_questions,
        latency_list=(df[end_time_header] - df[start_time_header])
        .sort_values()
        .to_list(),
        total_input_tokens_list=df[total_input_tokens_header].sort_values().to_list(),
        total_output_tokens_list=df[total_output_tokens_header].sort_values().to_list(),
    )


def find_and_extract_results(
    prefixes: list[str] = [BASELINE_PREFIX, RESULT_PREFIX],
    result_dir: Path = RESULT_DIR,
) -> list[ResultObject]:
    if not result_dir.is_dir():
        print(f"Error: '{result_dir}' is not a valid directory.")
        return []

    results = []
    for file_path in result_dir.iterdir():
        if (
            file_path.is_file()
            and (file_path.name).startswith(tuple(prefixes))
            and file_path.suffix == ".csv"
            and "split" in file_path.stem.split("_")
        ):
            results.append(create_result_object(file_path))

    return sorted(results)


def filter_result_factory(chunk_size: int, chunk_overlap: int):
    def filter_result(
        result: ResultObject, identifier: Optional[Identifier] = None
    ) -> bool:
        prefilter = (
            result.chunk_size == chunk_size and result.chunk_overlap == chunk_overlap
        )
        if identifier is None:
            return prefilter
        return (
            prefilter
            and result.run_type == identifier.run_type
            and result.provider == identifier.provider
            and result.model == identifier.model
            and result.num_frames_wiki_links == identifier.num_frames_wiki_links
        )

    return filter_result


def filter_model_result(
    result: ResultObject, model_identifier: ModelIdentifier
) -> bool:
    return (
        result.provider == model_identifier.provider
        and result.model == model_identifier.model
        and result.chunk_size == model_identifier.chunk_size
        and result.chunk_overlap == model_identifier.chunk_overlap
    )


def get_identifiers(
    results: list[ResultObject], filter_result=lambda _: True
) -> list[Identifier]:
    return sorted(
        list(
            {
                Identifier(
                    run_type=result.run_type,
                    provider=result.provider,
                    model=result.model,
                    num_frames_wiki_links=result.num_frames_wiki_links,
                )
                for result in results
                if filter_result(result)
            }
        )
    )


def get_model_identifiers(
    results: list[ResultObject], filter_result=lambda _: True
) -> list[ModelIdentifier]:
    return sorted(
        list(
            {
                ModelIdentifier(
                    provider=result.provider,
                    model=result.model,
                    chunk_size=result.chunk_size,
                    chunk_overlap=result.chunk_overlap,
                )
                for result in results
                if filter_result(result)
            }
        )
    )


def get_attributes_list(
    attribute: Literal[
        "num_docs",
        "num_questions",
        "accuracy",
        "latency_list",
        "total_input_tokens_list",
        "total_output_tokens_list",
    ],
    results: list[ResultObject],
    identifiers: list[Identifier],
    filter_result=lambda x, y: True,
):
    return {
        identifier: [
            getattr(result, attribute)
            for result in results
            if filter_result(result, identifier)
        ]
        for identifier in identifiers
    }


def get_models_list(
    results: list[ResultObject],
    model_identifiers: list[ModelIdentifier],
    filter_result=lambda x, y: True,
):
    return {
        model_identifier: [
            result for result in results if filter_result(result, model_identifier)
        ]
        for model_identifier in model_identifiers
    }


def plot_accuracy(
    results: list[ResultObject],
    dest: Path,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
):
    filter_result = filter_result_factory(chunk_size, chunk_overlap)
    identifiers = get_identifiers(results, filter_result)

    num_docs_list = get_attributes_list("num_docs", results, identifiers, filter_result)
    metric_list = get_attributes_list("accuracy", results, identifiers, filter_result)

    unique_num_docs = sorted(
        set(doc for docs in num_docs_list.values() for doc in docs)
    )

    x_indices = np.arange(len(unique_num_docs))
    bar_width = 0.8 / len(identifiers)

    plt.figure()

    for i, identifier in enumerate(identifiers):
        docs = num_docs_list[identifier]
        values = metric_list[identifier]

        value_map = dict(zip(docs, values))
        bar_values = [value_map.get(x, 0) for x in unique_num_docs]

        x_pos = x_indices + i * bar_width
        plt.bar(
            x_pos,
            bar_values,
            width=bar_width,
            label=str(identifier),
        )

    plt.xlabel("Number of Retrieved Documents")
    plt.ylabel("Accuracy")
    plt.xticks(x_indices + bar_width * (len(identifiers) - 1) / 2, unique_num_docs)
    plt.ylim(0, 1)
    plt.legend(
        handletextpad=0.5,
        labelspacing=0.2,
        borderaxespad=0.3,
    )
    plt.savefig(
        dest,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def _scatter_plot(
    x_vals: list[float],
    y_vals: list[float],
    labels: list[str],
    markers: list[str],
    colors: list[str],
    xlabel: str,
    ylabel: str,
    save_path: Path,
):
    plt.figure()
    for x, y, label, marker, color in zip(x_vals, y_vals, labels, markers, colors):
        plt.scatter(x, y, label=label, marker=marker, color=color, alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.legend(fontsize=11)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_color(result: ResultObject, color_map: dict[int, str]):
    if result.num_frames_wiki_links not in color_map:
        color_counter = len(color_map) % len(COLORS)
        color_map[result.num_frames_wiki_links] = COLORS[color_counter]
    color = color_map[result.num_frames_wiki_links]
    color_map[result.num_frames_wiki_links] = darken_hex_color(color)
    return color


def _generate_scatter_data(
    results: list[ResultObject],
    x_fn: Callable,
    y_fn: Callable,
    filter_fn: Callable = lambda r: True,
):
    x_vals, y_vals, labels, markers, colors = [], [], [], [], []
    color_map = {}
    use_labels = {}
    num_docs = defaultdict(list[int])
    for result in results:
        if not filter_fn(result):
            continue
        x_vals.append(x_fn(result))
        y_vals.append(y_fn(result))
        labels.append(result.to_label())
        num_docs[labels[-1]].append(result.num_docs)
        if result.num_frames_wiki_links not in use_labels:
            use_labels[result.num_frames_wiki_links] = len(labels) - 1
        colors.append(get_color(result, color_map))
        markers.append("x" if result.run_type == "baseline" else "o")

    labels = [
        (
            f"{label} ({','.join(map(str, num_docs[label]))} chunks)"
            if idx in use_labels.values()
            else ""
        )
        for idx, label in enumerate(labels)
    ]
    return x_vals, y_vals, labels, markers, colors


def plot_accuracy_vs_latency(results: list, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_identifiers = get_model_identifiers(results)
    models_list = get_models_list(results, model_identifiers, filter_model_result)
    for quantile in QUANTILES:
        (dest_dir / f"p{quantile}").mkdir(exist_ok=True)

        for model_identifier in model_identifiers:
            x_fn = lambda r: np.percentile(r.latency_list, quantile)
            y_fn = lambda r: r.accuracy
            x_vals, y_vals, labels, markers, colors = _generate_scatter_data(
                models_list[model_identifier], x_fn, y_fn
            )
            _scatter_plot(
                x_vals,
                y_vals,
                labels,
                markers,
                colors,
                xlabel=f"Latency P{quantile} in seconds",
                ylabel="Accuracy",
                save_path=dest_dir / f"p{quantile}" / f"{str(model_identifier)}.pdf",
            )


def _plot_individual_and_comparison(
    results: list[ResultObject],
    dest_dir: Path,
    data_getter,  # function: ResultObject -> list of x-values
    x_label: str,
):
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Plot individual CDFs
    for result in results:
        data = data_getter(result)
        cdf = np.arange(1, result.num_questions + 1) / result.num_questions

        plt.figure()
        plt.step(data, cdf, where="post")
        plt.xlabel(x_label)
        plt.ylabel("CDF")
        plt.ylim(0, 1)
        plt.savefig(
            dest_dir / f"{result.filename}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Plot comparisons
    model_identifiers = get_model_identifiers(results)
    models_list = get_models_list(results, model_identifiers, filter_model_result)
    for model_identifier in model_identifiers:
        (dest_dir / "comparison").mkdir(exist_ok=True)
        plt.figure()
        color_map = {}
        for result in models_list[model_identifier]:
            data = data_getter(result)
            cdf = np.arange(1, result.num_questions + 1) / result.num_questions
            plt.step(
                data,
                cdf,
                where="post",
                color=get_color(result, color_map),
                label=str(result),
            )

        plt.xlabel(x_label)
        plt.ylabel("CDF")
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(
            dest_dir / "comparison" / f"{str(model_identifier)}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def plot_latency(
    results: list[ResultObject],
    dest_dir: Path,
):
    _plot_individual_and_comparison(
        results=results,
        dest_dir=dest_dir,
        data_getter=lambda r: r.latency_list,
        x_label="Latency in seconds",
    )


def plot_total_input_tokens(
    results: list[ResultObject],
    dest_dir: Path,
):
    _plot_individual_and_comparison(
        results=results,
        dest_dir=dest_dir,
        data_getter=lambda r: r.total_input_tokens_list,
        x_label="Total Input Tokens",
    )


def plot_total_output_tokens(
    results: list[ResultObject],
    dest_dir: Path,
):
    _plot_individual_and_comparison(
        results=results,
        dest_dir=dest_dir,
        data_getter=lambda r: r.total_output_tokens_list,
        x_label="Total Output Tokens",
    )


def main():
    results = find_and_extract_results()
    plot_accuracy(results, SEQUENTIAL_PLOT_DIR / "accuracy.pdf")
    plot_latency(results, SEQUENTIAL_PLOT_DIR / "latency")
    plot_accuracy_vs_latency(results, SEQUENTIAL_PLOT_DIR / "accuracy_vs_latency")
    plot_total_input_tokens(results, SEQUENTIAL_PLOT_DIR / "total_input_tokens")
    plot_total_output_tokens(results, SEQUENTIAL_PLOT_DIR / "total_output_tokens")


if __name__ == "__main__":
    main()
