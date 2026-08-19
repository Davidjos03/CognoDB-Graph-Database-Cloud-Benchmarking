"""Allow ``python -m benchmark`` as well as the ``gdbbench`` script."""

import sys

from benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
