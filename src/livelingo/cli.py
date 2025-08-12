"""Command line interface for LiveLingo client."""

import argparse
from typing import Sequence, Optional

from .client import run_client


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for the client."""
    ap = argparse.ArgumentParser(
        description="System-audio loopback -> TCP client (prints subtitles & optional SRT)"
    )
    ap.add_argument("host", nargs="?", default="127.0.0.1")
    ap.add_argument("port", nargs="?", type=int, default=43007)
    ap.add_argument("--sr", type=int, default=16000, help="sample rate (default: 16000)")
    ap.add_argument("--block", type=int, default=1024, help="frames per block (default: 1024=~64ms)")
    ap.add_argument("--srt", type=str, default=None, help="save subtitles to this .srt file")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_client(args.host, args.port, args.sr, args.block, args.srt)


if __name__ == "__main__":
    main()

