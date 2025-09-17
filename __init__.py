"""
FACEScore: Fourier Analysis of Cross-Entropy for evaluating open-ended natural language generation.

This package provides tools for evaluating natural language generation models using
spectral features of text surprisal through Fourier analysis.
"""

from .face_score import FACEScorer
from .fft_utils import FFTProcessor
from .metrics import cal_metrics

__version__ = "1.0.0"
__author__ = "FACEScore Team"
__email__ = ""

__all__ = [
    "FACEScorer",
    "FFTProcessor", 
    "cal_metrics",
]
