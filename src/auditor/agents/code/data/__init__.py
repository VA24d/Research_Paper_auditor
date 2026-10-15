"""Data analysis mode: what a dataset's data files actually hold (decision D4 data classes).

`splits` maps files to splits, `readers` streams their rows, `analysis` counts rows and
labels and finds leakage, and `report` ties these to a snapshot and caches the result.
"""
