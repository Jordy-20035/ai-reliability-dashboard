"""
Data drift detection: compare production or monitoring data to a training baseline.

Statistical methods: PSI, Kolmogorov-Smirnov, Chi-square (per feature type).
"""

from . import baseline, detectors

__all__ = ["baseline", "detectors"]
