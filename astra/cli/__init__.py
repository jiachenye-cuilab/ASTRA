"""Command-line entry point; load only the selected command's dependencies."""
import argparse
from importlib import import_module
import sys

from astra.assets import ROOT
from astra import __version__


COMMANDS = {
    "prepare-benchmark": ("prepare_benchmark", "Prepare native HD16 or pseudo-Visium inputs and separate measured reference"),
    "benchmark": ("benchmark", "Run frozen ASTRA and score measured 8um reference"),
    "evaluate": ("evaluate", "Score serialized ASTRA predictions against measured 8um reference"),
    "prepare-visium": ("prepare_visium", "Align 10x Visium counts and registered positions for 160/144 inference"),
    "configure-training": ("configure_training", "Create a separate training workspace for user Visium HD sections"),
    "predict": ("predict", "Reconstruct one field of view"),
    "predict-section": ("predict_section", "Reconstruct and stitch a whole section"),
    "prepare-input": ("prepare_input", "Prepare a registered field of view"),
    "prepare-features": ("prepare_features", "Cache frozen UNI features"),
    "prepare-section": ("prepare_section", "Prepare a whole-section cache"),
    "prepare-training": ("prepare_training", "Prepare base-training data"),
    "train": ("train", "Train from scratch or resume a checkpoint"),
    "fine-tune": ("fine_tune", "Adapt to one section using coarse observations"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m astra", description="ASTRA spatial transcriptomic reconstruction")
    parser.add_argument("--version", action="version",
                        version=f"{__version__} (model snapshot {(ROOT / 'VERSION').read_text(encoding='utf-8').strip()})")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (_, description) in COMMANDS.items():
        subparsers.add_parser(name, help=description, add_help=False)
    if argv and argv[0] in COMMANDS:
        name = argv[0]
        command = import_module("." + COMMANDS[name][0], __name__)
        try:
            return command.main(argv[1:], prog=f"{parser.prog} {name}")
        except Exception as error:
            # Preserve actionable tracebacks, with one concise CUDA-memory hint.
            import torch
            if not isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            raise SystemExit("CUDA memory budget exceeded; lower the batch size or use --device cpu.") from None
    parser.parse_args(argv)
