#!/usr/bin/env python3
"""Simple development server for the nozzle web interface.

Serves the web/ directory and makes nozzle/ source files available
at /nozzle/nozzle/*.py so Pyodide can fetch them.

Usage:
    python web/serve.py [--port 8080]
    # Then open http://localhost:8080
"""

import argparse
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class NozzleHandler(SimpleHTTPRequestHandler):
    """Serves web/ as root and nozzle/ source at /nozzle/nozzle/."""

    def __init__(self, *args, project_root=None, **kwargs):
        self.project_root = project_root
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        # Serve /nozzle/nozzle/*.py from the project's nozzle/ directory
        if path.startswith('/nozzle/nozzle/'):
            rel = path[len('/nozzle/'):]
            return str(self.project_root / rel)
        # Everything else from web/
        return str(self.project_root / 'web' / path.lstrip('/'))

    def end_headers(self):
        # CORS and caching headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description='Nozzle web dev server')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    def handler(*a, **kw):
        return NozzleHandler(*a, project_root=project_root, **kw)

    server = HTTPServer(('localhost', args.port), handler)
    print(f'Serving nozzle web interface at http://localhost:{args.port}')
    print(f'Project root: {project_root}')
    print('Press Ctrl+C to stop.')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
