from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    handler = lambda *args, **kwargs: Handler(*args, directory=root, **kwargs)
    print('Kraken Lite: http://127.0.0.1:8771/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8771), handler).serve_forever()
