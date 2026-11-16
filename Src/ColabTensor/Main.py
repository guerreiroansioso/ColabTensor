"""CLI entry point: provision a G4 Colab session serving both models."""

import argparse
import sys

from ColabTensor.ColabSession import ColabSession, ColabSessionError
from ColabTensor.Provisioner import Provisioner

ONLY_ALIASES = {
    "gpt-oss": "gpt-oss-120b",
    "decider": "pplx-decider-v1.1",
    "gpt-oss-120b": "gpt-oss-120b",
    "pplx-decider-v1.1": "pplx-decider-v1.1",
    "both": "both",
}

EPILOG = """\
examples:
  python -m ColabTensor.Main --session colabtensor --gpu G4
  python -m ColabTensor.Main --session colabtensor --only gpt-oss
  python -m ColabTensor.Main --session colabtensor --only decider
  python -m ColabTensor.Main --session colabtensor --dry-run
  python -m ColabTensor.Main --session colabtensor --stop

notes:
  Both models are served by llama-server (llama.cpp) as Q4 GGUF.
  gpt-oss-120b: unsloth/gpt-oss-120b-GGUF:Q4_K_M (~63 GB).
  pplx-decider-v1.1: turintech/pplx-decider-27B-Artemis-Q4_K_S (~15 GB),
  based on pplx-decider-v1.1-27b (protocol v1).
  Both models together need ~78 GB of GPU memory; a single 96 GB G4 fits.
"""


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colabtensor",
        description="Serve gpt-oss-120b and pplx-decider-v1.1 (Q4 GGUF) on "
        "a Colab G4 via llama.cpp, exposed with localtunnel.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session", default="colabtensor", help="session name")
    parser.add_argument(
        "--gpu",
        default="G4",
        help="GPU variant (default: G4). A100 implies --high-mem (80 GB).",
    )
    parser.add_argument(
        "--only",
        choices=list(ONLY_ALIASES),
        default="both",
        help="which models to serve (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without executing them",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="stop the session and exit",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=3600,
        help="seconds to wait for each model to become healthy",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    try:
        if args.stop:
            ColabSession(args.session).stop()
            print(f"[colabtensor] stopped session {args.session}")
            return 0
        provisioner = Provisioner(
            sessionName=args.session,
            gpu=args.gpu,
            only=ONLY_ALIASES[args.only],
            dryRun=args.dry_run,
            healthTimeoutSec=args.health_timeout,
        )
        provisioner.run()
    except ColabSessionError as error:
        print(f"[colabtensor] error: {error}", file=sys.stderr)
        if "colab new" in str(error):
            print(
                "[colabtensor] hint: the GPU assignment was rejected by "
                "Colab. G4 is tier-gated and capacity-limited; check your "
                "Colab subscription entitlement, retry later, or pick "
                "another GPU with --gpu (both models need ~78 GB, so only "
                "H100/A100 80 GB fit).",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
