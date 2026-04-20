"""luxe-digit — ML-augmented detector digitization for the LUXE GBP.

This package builds on the SimpleDigitizerGP pipeline by Pietro Grutta
(INFN Padova / LUXE collaboration) and adds:

    * A PyTorch-based ML feature extractor (`luxedigit.ml_extractor`)
    * A CLI entry point (`luxedigit.cli`)
    * PyROOT-free I/O via uproot (`luxedigit.io`)

The core digitization classes (`readFromMc`, `frontend`, `featureExtractor`)
are preserved from the upstream project and remain the authoritative
reference implementation of the physics.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "YOUR_FULL_NAME"
__license__ = "GPL-3.0-or-later"

__all__ = ["__version__"]
