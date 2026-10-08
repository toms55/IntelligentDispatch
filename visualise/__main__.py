"""Plot a results/<REGION>.md file.

    uv run python -m visualise results/NSW1.md
    uv run python -m visualise results/NSW1.md --out NSW1.png
    uv run python -m visualise results/NSW1.md --show
"""
import argparse
from pathlib import Path

from visualise.parse import loadResultsFile
from visualise.plots import buildFigure, saveFigure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="a results/<REGION>.md file")
    parser.add_argument("--out", type=Path, metavar="PATH",
                         help="where to save the figure; defaults to the input path "
                              "with a .png suffix")
    parser.add_argument("--show", action="store_true",
                         help="display the figure in a window instead of saving it")
    args = parser.parse_args(argv)

    sections = loadResultsFile(args.path)
    region = sections[0].region
    algo = sections[0].algo

    if args.show:
        import matplotlib.pyplot as plt
        buildFigure(sections, region, algo)
        plt.show()
        return 0

    destination = args.out or args.path.with_suffix(".png")
    saveFigure(sections, region, algo, destination)
    print(f"wrote          {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
