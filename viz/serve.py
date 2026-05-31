"""
Tiny static server so the viz pages can fetch JSONL live (file:// blocks fetch).

No database, no build step: the pages read data/*.jsonl and results/history.jsonl directly,
so just keep this running and refresh the browser after generating data or running an eval.

Usage:  python viz/serve.py            # serves project root on :8000, opens results.html
        python viz/serve.py --port 9000 --no-open
"""
import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        base = f"http://127.0.0.1:{args.port}"
        print(f"Serving {ROOT} at {base}")
        print(f"  dashboard : {base}/viz/index.html  (tabs: Results · Dataset · Versions · Findings)")
        print("              domain sub-tabs (All · IT · Legal · Marketing) filter every tab")
        print("Ctrl-C to stop.")
        if not args.no_open:
            threading.Timer(0.5, lambda: webbrowser.open(f"{base}/viz/index.html")).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
