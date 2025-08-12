"""Entry point for LiveLingo."""

import os
import sys

# Ensure src is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from livelingo.cli import build_parser
from livelingo.client import run_client


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Run the audio client in a background thread
    import threading
    client_thread = threading.Thread(
        target=run_client,
        args=(args.host, args.port, args.sr, args.block, args.srt),
        daemon=True,
    )
    client_thread.start()

    # Launch the ImGui overlay UI in the main thread
    from livelingo.ui import run_ui  # Lazy import to avoid OpenGL requirement for CLI

    run_ui()


if __name__ == "__main__":
    main()
