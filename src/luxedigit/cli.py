"""Command-line interface for luxe-digit.

Exposes two primary verbs:

* ``luxedigit train`` — train the ML feature extractor on labeled profiles.
* ``luxedigit predict`` — run inference, replacing the Gaussian+constant fit.

Plus a diagnostic ``luxedigit info`` for reporting installed versions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from luxedigit import __version__

log = logging.getLogger("luxedigit.cli")


# ─────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────


def cmd_info(_: argparse.Namespace) -> int:
    import platform

    print(f"luxe-digit  {__version__}")
    print(f"python      {platform.python_version()}  ({platform.platform()})")

    try:
        import torch

        print(f"torch       {torch.__version__}  (cuda: {torch.cuda.is_available()})")
    except ImportError:
        print("torch       not installed  (install with `pip install luxe-digit[ml]`)")

    try:
        import uproot

        print(f"uproot      {uproot.__version__}")
    except ImportError:
        print("uproot      not installed")

    try:
        import ROOT

        print(f"ROOT        {ROOT.gROOT.GetVersion()}")
    except ImportError:
        print("ROOT        not installed  (Docker image provides it)")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the ML feature extractor."""
    from luxedigit.ml_extractor import (
        MLExtractor,
        MLExtractorConfig,
        synthesize_training_set,
    )

    log.info("Loading training data…")
    if args.synthetic:
        log.warning(
            "Using synthetic training data — set --input to a .npz file with "
            "'profiles' and 'labels' arrays for real training."
        )
        profiles, labels = synthesize_training_set(
            n_samples=args.synthetic_n,
            n_strips=args.n_strips,
        )
    else:
        if args.input is None:
            log.error("--input is required unless --synthetic is passed")
            return 2
        data = np.load(args.input)
        profiles, labels = data["profiles"], data["labels"]

    log.info("Training on %d samples, %d strips", profiles.shape[0], profiles.shape[1])
    cfg = MLExtractorConfig(
        n_strips=profiles.shape[1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    extractor = MLExtractor(cfg)
    history = extractor.fit(profiles, labels)

    log.info(
        "Training complete. Final val loss: %.5f",
        history["val_loss"][-1] if history["val_loss"] else float("nan"),
    )
    out_path = Path(args.output)
    extractor.save(out_path)
    log.info("Saved model → %s", out_path)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Run inference on new profiles."""
    from luxedigit.ml_extractor import MLExtractor

    log.info("Loading model from %s", args.model)
    extractor = MLExtractor.load(args.model)

    data = np.load(args.input)
    profiles = data["profiles"] if "profiles" in data else data["arr_0"]
    log.info("Predicting on %d profiles (mc_samples=%d)", profiles.shape[0], args.mc_samples)

    result = extractor.predict(profiles, mc_samples=args.mc_samples)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        params=result["params"],
        std=result["std"],
        labels=np.array(result["labels"]),
    )
    log.info("Saved predictions → %s", out_path)
    return 0


# ─────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="luxedigit",
        description="ML-augmented detector digitization pipeline for the LUXE GBP.",
    )
    p.add_argument("--version", action="version", version=f"luxe-digit {__version__}")
    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)"
    )

    sub = p.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="Report installed versions of dependencies")
    p_info.set_defaults(func=cmd_info)

    # train
    p_train = sub.add_parser("train", help="Train the ML feature extractor")
    p_train.add_argument("--input", type=Path, help=".npz file with 'profiles' and 'labels' arrays")
    p_train.add_argument("--output", type=Path, required=True, help="Output .pt model file")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--n-strips", type=int, default=64)
    p_train.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic Gaussian profiles instead of --input (smoke-test mode)",
    )
    p_train.add_argument(
        "--synthetic-n",
        type=int,
        default=5000,
        help="Number of synthetic samples (only used with --synthetic)",
    )
    p_train.set_defaults(func=cmd_train)

    # predict
    p_pred = sub.add_parser("predict", help="Run inference with a trained model")
    p_pred.add_argument("--input", type=Path, required=True, help=".npz file with 'profiles'")
    p_pred.add_argument("--model", type=Path, required=True, help="Saved .pt model file")
    p_pred.add_argument("--output", type=Path, required=True, help="Output .npz predictions")
    p_pred.add_argument(
        "--mc-samples", type=int, default=30, help="Number of MC-dropout forward passes"
    )
    p_pred.set_defaults(func=cmd_predict)

    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.error("Unhandled error: %s", exc, exc_info=args.verbose >= 2)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
