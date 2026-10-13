"""Consistent typography, immutable figure provenance and deterministic exports."""

import hashlib
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import POLICIES


COLORS = dict(zip(POLICIES, ("#28734f", "#2864ad", "#db881b", "#8266aa", "#707070", "#b74d55")))
MODEL_COLORS = {"reference_5250": COLORS[POLICIES[3]], "original_10000_best": COLORS[POLICIES[2]], "augmented_10000_best": COLORS[POLICIES[1]]}


def configure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "grid.alpha": .2, "svg.hashsalt": "RL_LAB_IA-stage-6.5", "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white"})


class FigureWriter:
    def __init__(self, output, source_hashes):
        self.output = output
        self.source_hashes = source_hashes
        self.figures = []

    def save(self, fig, name, role, caption, sources, plotted_data):
        # Reserve a dedicated footer area, rather than putting provenance on axes.
        fig.text(.01, .015, caption, ha="left", va="bottom", fontsize=7, color="#444444")
        fig.tight_layout(rect=(0, .095, 1, 1))
        record = {"name": name, "dataset_role": role, "caption": caption,
            "sources": {source: self.source_hashes[source] for source in sources},
            "plotted_data": plotted_data, "files": {}, "deterministic_export_verified": True}
        for extension, metadata in (("png", {"Software": "RL_LAB_IA Stage 6.5"}),
                                    ("svg", {"Date": None, "Creator": "RL_LAB_IA Stage 6.5"})):
            def render():
                stream = BytesIO()
                fig.savefig(stream, format=extension, dpi=300, metadata=metadata)
                return stream.getvalue()
            payload = render()
            if payload != render():
                raise AssertionError("Figure export is not deterministic: " + name + "." + extension)
            path = self.output / (name + "." + extension)
            path.write_bytes(payload)
            record["files"][extension] = {"path": "figures/" + path.name,
                "sha256": hashlib.sha256(payload).hexdigest(), "dpi": 300 if extension == "png" else None}
        plt.close(fig)
        self.figures.append(record)
        print("Saved " + name + " (PNG 300 DPI + SVG)", flush=True)


def rolling_mean(values, window):
    """Trailing arithmetic mean with a shorter prefix; NaNs preserve missing data."""
    import numpy as np
    array = np.asarray(values, dtype=float)
    if window < 1 or array.ndim != 1:
        raise ValueError("Rolling mean requires a positive window and a one-dimensional series")
    finite = np.isfinite(array)
    sums = np.concatenate(([0.], np.cumsum(np.where(finite, array, 0.))))
    counts = np.concatenate(([0], np.cumsum(finite)))
    ends = np.arange(1, len(array) + 1)
    starts = np.maximum(0, ends - window)
    denominator = counts[ends] - counts[starts]
    output = np.full(len(array), np.nan)
    np.divide(sums[ends] - sums[starts], denominator, out=output, where=denominator > 0)
    return output.tolist()
