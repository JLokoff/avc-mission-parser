"""V3 multi-model entry point. See README.md. Uses the shared implementation."""
import sys
from compare import main

if __name__ == "__main__":
    main(['eval'] + sys.argv[1:])
