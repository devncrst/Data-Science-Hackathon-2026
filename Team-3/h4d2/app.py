"""h4d2 bench app: the results dashboard plus a model bench, served locally.

    python -m h4d2.app            # results: http://127.0.0.1:8765   model bench: http://127.0.0.1:8765/bench
    python -m h4d2.app --port 9000 --no-browser

Drop a model .pkl (and optionally a new city's parquet) on the page; simulations run in the background and the
dashboard shows progress and scores.  Env: H4D_DATA_DIR (organiser parquet folder), H4D2_OUT (artifacts),
H4D2_BENCH_HOME (where uploads and runs are kept, default ~/h4d2_bench).
Binds to 127.0.0.1 only.  Only upload .pkl files you trust: unpickling runs code.
"""
from __future__ import annotations
import argparse, json, threading, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from . import bench, dashboard_live


def page(view="results"):
    """view "results": our model's results dashboard; view "bench": the model bench on its own page."""
    doc = dashboard_live.page()
    flag = '<script>window.H4D2_APP=true;window.H4D2_VIEW=%s;</script>' % json.dumps(view)
    return doc.replace('<script id="payload"', flag + '<script id="payload"', 1)


class Handler(BaseHTTPRequestHandler):
    server_version = "h4d2-bench"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, e, code=400):
        self._json(dict(error=str(e)), code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, page("results").encode(), "text/html; charset=utf-8")
            if u.path in ("/bench", "/bench/"):
                return self._send(200, page("bench").encode(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json(dict(datasets=bench.list_datasets(), models=bench.list_models(),
                                       leaderboard=bench.leaderboard(), shots=list(bench.SHOTS)))
            if u.path == "/api/job":
                j = dict(bench.JOBS.get(q.get("id"), dict(status="unknown")))
                return self._json(j)
            if u.path == "/api/result":
                return self._json(bench.get_run(q["id"]))
            return self._err("not found", 404)
        except Exception as e:
            return self._err(e)

    def do_POST(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/dataset":
                return self._json(bench.add_dataset(q.get("name", "city"), int(q.get("event", 1945)), self._body()))
            if u.path == "/api/model":
                return self._json(bench.add_model(q.get("name", "model"), self._body()))
            if u.path == "/api/run":
                a = json.loads(self._body() or b"{}")
                jid = bench.submit(a["dataset"], a["models"], a.get("shots", bench.SHOTS),
                                   int(a.get("trials", 10)), int(a.get("seed", 42)), a.get("modes"))
                return self._json(dict(id=jid))
            if u.path == "/api/parity":
                a = json.loads(self._body() or b"{}")
                return self._json(dict(id=bench.submit_parity(a["dataset"], a["model"])))
            return self._err("not found", 404)
        except Exception as e:
            return self._err(e)


def main():
    ap = argparse.ArgumentParser(prog="h4d2.app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = "http://127.0.0.1:%d" % a.port
    print("h4d2 results: %s   model bench: %s/bench   (Ctrl+C to stop)" % (url, url), flush=True)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
