"""Entry point for LiveLingo."""

import os
import sys

# Ensure src is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from livelingo.cli import build_parser
from livelingo.client import run_client


def main() -> None:
    parser = build_parser()
    parser.add_argument("--gui", action="store_true", help="run ImGui interface instead of CLI")
    args = parser.parse_args()
    if args.gui:
        from livelingo.ui import run_ui  # Lazy import to avoid OpenGL requirement for CLI

        run_ui()
    else:
        run_client(args.host, args.port, args.sr, args.block, args.srt)


if __name__ == "__main__":
    main()
