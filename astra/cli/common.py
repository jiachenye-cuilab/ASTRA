"""Shared argument validation for ASTRA commands."""
import argparse
import math


def positive_gib(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("GPU memory budget must be finite and positive")
    return value
