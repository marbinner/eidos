#!/usr/bin/env python3
"""Serve the browser build of eidos.

Differences from `python -m http.server` that matter here:
  * sends the pre-compressed `.gz` copy of a file when there is one (the program is
    about 8 MB raw and about 3 MB compressed, which is most of the loading time when
    the connection is an SSH tunnel);
  * every response is revalidated (`Cache-Control: no-cache` with an ETag), so a
    reload costs one tiny request when nothing changed and picks up a rebuild at once;
  * serves `.wasm` with the MIME type browsers need for streaming compilation.

usage: serve.py <directory> [port]
"""
import http.server
import os
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".webmanifest": "application/manifest+json",
        ".wasm": "application/wasm",
        ".js": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path.split("?", 1)[0].split("#", 1)[0])
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        packed = path + ".gz"
        wants_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        if not (wants_gzip and os.path.isfile(path) and os.path.isfile(packed)
                and os.path.getmtime(packed) >= os.path.getmtime(path)):
            return super().send_head()
        stat = os.stat(packed)
        etag = f'"{int(stat.st_mtime)}-{stat.st_size}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return None
        f = open(packed, "rb")
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("ETag", etag)
        self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        return f

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    os.chdir(directory)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"eidos: http://localhost:{port}   (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
