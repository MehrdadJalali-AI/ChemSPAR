"""Local helper for Figure 1: serves figures/fig1_build and saves PNGs POSTed by render.html (data URLs).
Usage: python fig1_render_server.py 8765   (localhost only)"""
import base64
import http.server
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "figures/fig1_build"


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(D), **k)

    def do_POST(self):
        name = self.path.split("name=")[-1].replace("/", "")
        data = self.rfile.read(int(self.headers["Content-Length"])).decode()
        (D / f"{name}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
