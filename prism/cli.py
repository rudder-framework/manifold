"""
PRISM CLI

FULL COMPUTE. NO EXCEPTIONS.

Usage:
    python -m prism manifest.yaml
    python -m prism observations.parquet

That's it. Everything runs. 100%.
"""

import sys
from pathlib import Path


def main():
    """Run PRISM. Full compute. No exceptions."""

    # Import here to avoid circular imports
    from prism.runner import run, DATA_DIR

    # No arguments = use canonical data/manifest.yaml
    if len(sys.argv) < 2:
        result = run()  # Uses DATA_DIR/manifest.yaml
        print(f"\nResults written to: {result.get('output_dir', 'unknown')}")
        return 0

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Determine input type
    if input_path.suffix in ['.yaml', '.yml', '.json']:
        # Manifest file - run directly
        result = run(input_path)
    elif input_path.suffix == '.parquet':
        # Direct parquet - create temp manifest and run
        import json
        import tempfile

        output_dir = input_path.parent

        manifest = {
            'dataset': {'name': input_path.stem},
            'data': {
                'observations_path': str(input_path.resolve()),
            },
            'prism': {
                'window_size': 100,
                'stride': 50,
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest, f)
            temp_path = Path(f.name)

        try:
            result = run(temp_path)
        finally:
            temp_path.unlink()
    else:
        print(f"Error: Unknown file type: {input_path.suffix}")
        print("Expected: .yaml, .yml, .json, or .parquet")
        sys.exit(1)

    print(f"\nResults written to: {result.get('output_dir', 'unknown')}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
