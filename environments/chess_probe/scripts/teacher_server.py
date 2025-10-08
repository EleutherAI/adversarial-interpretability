"""Minimal HTTP server exposing ActionValueTeacher endpoints.

Endpoints:
- POST /get_hidden_states {"fen": str, "move": str} -> {"hidden": [float]}
- POST /get_move_win_probs {"fen": str} -> {"moves": [str], "win_probs": [float]}
- GET  /meta -> {"model_size": str, "embedding_dim": int, "num_layers": int}
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.append(str(_VENDOR_ROOT))


def create_app(model_size: str, hidden_layer_idx: str | None, device: str):
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.pop("JAX_PLATFORMS", None)

    from environments.chess_probe.models.teacher_wrapper import ActionValueTeacher  # noqa: E402

    # Parse hidden_layer_idx which may be comma-separated list or integer/None
    parsed_idx = None
    if hidden_layer_idx is not None:
        s = str(hidden_layer_idx)
        if "," in s:
            try:
                parsed_idx = [int(x.strip()) for x in s.split(",") if x.strip()]
            except Exception:
                parsed_idx = None
        else:
            try:
                parsed_idx = int(s)
            except Exception:
                parsed_idx = None
    teacher = ActionValueTeacher(
        model_size=model_size,
        checkpoint_step=-1,
        use_ema=True,
        hidden_layer_idx=parsed_idx,
    )

    # No behavioral cloning model; server only proxies action-value teacher

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/meta":
                self._json(200, {
                    "model_size": model_size,
                    "embedding_dim": teacher.embedding_dim,
                    "num_layers": teacher.num_layers,
                })
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                self._json(400, {"error": "invalid json"})
                return

            try:
                if path == "/get_hidden_states":
                    fen = data["fen"]
                    move = data["move"]
                    tensor = teacher.get_hidden_states(fen, move)
                    self._json(200, {"hidden": tensor.detach().cpu().tolist()})
                    return
                if path == "/get_hidden_states_batch":
                    fens = data["fens"]
                    moves = data["moves"]
                    if len(fens) != len(moves):
                        self._json(400, {"error": "fens and moves length mismatch"})
                        return
                    tensor = teacher.get_hidden_states_batch(fens, moves)
                    self._json(200, {"hidden": tensor.detach().cpu().tolist()})
                    return
                if path == "/get_move_win_probs":
                    fen = data["fen"]
                    moves, probs = teacher.get_move_win_probs(fen)
                    self._json(200, {"moves": moves, "win_probs": probs.tolist()})
                    return
            except Exception as e:
                self._json(500, {"error": str(e)})
                return

            self._json(404, {"error": "not found"})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_size", type=str, default="270M", choices=["9M", "136M", "270M"]) 
    parser.add_argument("--teacher_hidden_layer_idx", type=str, default=None,
                        help="Teacher layer index or comma-separated list (e.g., -1 or '2,4,6')")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"]) 
    args = parser.parse_args()

    handler = create_app(args.model_size, args.teacher_hidden_layer_idx, args.device)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Teacher server listening on http://{args.host}:{args.port} (model={args.model_size}, device={args.device})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()


