#!/usr/bin/env python3
"""Build the static web directory for GitHub Pages deployment.

Copies nozzle/*.py source files into web/nozzle/ so Pyodide can
fetch them via relative URLs from a static file host.

Usage:
    python web/build.py
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / 'web'
SRC_DIR = PROJECT_ROOT / 'nozzle'
DEST_DIR = WEB_DIR / 'nozzle'

MODULES = ['__init__', 'gas', 'kernel', 'moc', 'contours', 'analysis']


def main():
    DEST_DIR.mkdir(exist_ok=True)

    copied = []
    for mod in MODULES:
        src = SRC_DIR / f'{mod}.py'
        dst = DEST_DIR / f'{mod}.py'
        if not src.exists():
            print(f'  SKIP {mod}.py (not found)')
            continue
        shutil.copy2(src, dst)
        copied.append(mod)

    print(f'Copied {len(copied)} modules to {DEST_DIR.relative_to(PROJECT_ROOT)}/')
    for m in copied:
        print(f'  {m}.py')

    # Write .nojekyll so GitHub Pages doesn't filter underscores
    nojekyll = WEB_DIR / '.nojekyll'
    nojekyll.touch()
    print(f'Wrote {nojekyll.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
