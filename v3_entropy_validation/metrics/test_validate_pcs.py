"""
test_validate_pcs.py

Unit tests for validating the consistency and correctness of PCS/TDS/t-SNE analysis results.

This module performs automated checks to ensure that:
- The output CSV file 'tds_vs_epsilon.csv' is generated and contains valid data.
- The computed Trajectory Discrimination Scores (TDS) are finite and non-negative.
- The ε (epsilon) values in the dataset increase monotonically, ensuring parameter sweep consistency.

To run all tests:
    pytest test_validate_pcs.py -v

Author: Vladimir Khomyakov
Project: cognitive-observer-simulation / v3_entropy_validation
Date: 2025-06
"""

import pandas as pd
import os
import pytest

CSV_PATH = "verification_plots/tds_vs_epsilon.csv"

def test_tds_output_exists():
    """
    Ensure the CSV file containing TDS results exists.
    """
    assert os.path.exists(CSV_PATH), f"Expected file not found: {CSV_PATH}"

def test_tds_values_reasonable():
    """
    Verify that all TDS values are strictly positive.
    """
    df = pd.read_csv(CSV_PATH)

    assert "tds" in df.columns, (
        f"Column 'tds' not found in CSV. Available columns: {df.columns.tolist()}"
    )

    invalid_values = df[df["tds"] <= 0]

    assert invalid_values.empty, (
        f"Found non-positive TDS values:\n{invalid_values}"
    )

def test_epsilons_monotonic():
    """
    Verify that epsilon values increase strictly monotonically.
    """
    df = pd.read_csv(CSV_PATH)

    assert "epsilon" in df.columns, (
        f"Column 'epsilon' not found in CSV. Available columns: {df.columns.tolist()}"
    )

    eps = df["epsilon"].values
    diffs = [b - a for a, b in zip(eps[:-1], eps[1:])]
    non_increasing = [i for i, d in enumerate(diffs) if d <= 0]

    assert not non_increasing, (
        f"Epsilon values are not strictly increasing at indices: {non_increasing}, "
        f"differences: {diffs}"
    )
