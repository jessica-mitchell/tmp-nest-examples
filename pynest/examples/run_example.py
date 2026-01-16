#!/usr/bin/env python3
"""
Wrapper script for running NEST examples in CI.

This script:
1. Patches matplotlib.pyplot.show() to save figures instead of displaying
2. Runs the specified example
3. Collects all generated figures

Usage: python run_example.py <example_path> <output_dir>
"""

import sys
import os
import importlib.util
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Track figure counter for naming
_figure_counter = 0
_output_dir = None
_example_name = None


def _patched_show(*args, **kwargs):
    """Save all open figures instead of showing them."""
    global _figure_counter

    figs = [plt.figure(i) for i in plt.get_fignums()]
    for fig in figs:
        _figure_counter += 1
        filename = f"{_example_name}_fig{_figure_counter:02d}.png"
        filepath = os.path.join(_output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")

    # Close figures to free memory
    plt.close('all')


def run_example(example_path, output_dir):
    """Run an example and capture its figures."""
    global _output_dir, _example_name

    _output_dir = output_dir
    _example_name = os.path.splitext(os.path.basename(example_path))[0]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Patch plt.show
    original_show = plt.show
    plt.show = _patched_show

    # Also patch any imported matplotlib.pyplot
    import matplotlib.pyplot
    matplotlib.pyplot.show = _patched_show

    try:
        # Get the directory containing the example for imports
        example_dir = os.path.dirname(os.path.abspath(example_path))
        example_file = os.path.basename(example_path)

        # Add example directory to path for local imports
        if example_dir not in sys.path:
            sys.path.insert(0, example_dir)

        # Change to example directory
        original_cwd = os.getcwd()
        os.chdir(example_dir)

        # Load and execute the example
        spec = importlib.util.spec_from_file_location("example_module", example_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Save any remaining open figures
        if plt.get_fignums():
            _patched_show()

    finally:
        # Restore
        plt.show = original_show
        os.chdir(original_cwd)

    # List generated files
    outputs = os.listdir(output_dir)
    print(f"\nGenerated {len(outputs)} output files:")
    for f in sorted(outputs):
        print(f"  - {f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_example.py <example_path> <output_dir>")
        sys.exit(1)

    example_path = sys.argv[1]
    output_dir = sys.argv[2]

    run_example(example_path, output_dir)
