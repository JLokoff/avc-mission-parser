"""
Keep one trained model running on this computer so another program can send it mission
commands. Send a text command to the local /parse address and receive the six mission
fields or an error. The model stays loaded between requests; the server interprets
commands but never flies the drone or changes the mission itself.
"""
# Previously: serve.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission_parser import MissionParser

def make_handler(parser):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def respond(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                return self.respond(404, {'ok': False, 'error': 'not_found'})
            ready = parser.ready and parser.process.poll() is None
            self.respond(200 if ready else 503, {'ok': ready, 'model': parser.model, 'schema_version': 3})

        def do_POST(self):
            if self.path != '/parse':
                return self.respond(404, {'ok': False, 'error': 'not_found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 1 <= size <= 16384:
                    return self.respond(413, {'ok': False, 'error': 'body_size'})
                if self.headers.get_content_type() != 'application/json':
                    return self.respond(415, {'ok': False, 'error': 'expected_json'})
                obj = json.loads(self.rfile.read(size))
                if not isinstance(obj, dict) or set(obj) != {'text'}:
                    raise ValueError('Expected exactly {"text": "human command"}')
                result = parser.parse(obj['text'])
                self.respond(200 if result['ok'] else 422, result)
            except (ValueError, UnicodeError) as exc:
                self.respond(400, {'ok': False, 'error': 'invalid_request', 'detail': str(exc)})
            except Exception:
                self.respond(503, {'ok': False, 'error': 'inference_unavailable', 'intent': None})
    return Handler

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['granite350', 'lfm350', 'functiongemma'], default='granite350')
    p.add_argument('--quantization', choices=['Q8_0', 'Q6_K', 'Q4_K_M'], default='Q8_0')
    p.add_argument('--gguf', help='Optional explicit GGUF path; matching .json sidecar is required')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--gpu-layers', type=int, default=99, help='Use 0 for CPU-only inference')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--cache-prefix', action=argparse.BooleanOptionalAction, default=None,
                   help='Default: enabled only for Granite; other models are experimental')
    a = p.parse_args()
    if not 1 <= a.port <= 65535:
        p.error('Invalid port')
    with MissionParser(a.model, a.quantization, a.gguf, a.gpu_layers, a.threads, a.cache_prefix) as parser:
        with HTTPServer(('127.0.0.1', a.port), make_handler(parser)) as server:
            print(f'Ready at http://127.0.0.1:{a.port}; Ctrl+C to stop', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass

if __name__ == '__main__':
    main()
