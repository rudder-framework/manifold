"""
PRISM package entry point.

Allows running: python -m prism [command] [args]
"""

from prism.entry_points.pipeline import main

if __name__ == '__main__':
    main()
