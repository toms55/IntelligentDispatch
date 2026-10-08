"""Plot a results/<REGION>.md file.

    uv run python -m visualise results/NSW1.md
    uv run python -m visualise results/NSW1.md --out NSW1.png
    uv run python -m visualise results/NSW1.md --show

    uv run python -m visualise results/NSW1.md --best-vs-oracle \\
        --train-start 2024-01-01T04:00 --train-end 2025-01-01T04:00
"""
import argparse
from pathlib import Path

from visualise.parse import loadResultsFile
from visualise.plots import buildBestSeedFigure, buildFigure, saveBestSeedFigure, saveFigure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="a results/<REGION>.md file")
    parser.add_argument("--out", type=Path, metavar="PATH",
                         help="where to save the figure; defaults to the input path "
                              "with a .png suffix")
    parser.add_argument("--show", action="store_true",
                         help="display the figure in a window instead of saving it")
    parser.add_argument("--best-vs-oracle", action="store_true",
                         help="plot just the single best seed against the oracle, "
                              "instead of the default three-panel overview")
    parser.add_argument("--source", metavar="TEXT",
                         help="with --best-vs-oracle, pick the best seed from the "
                              "section whose heading contains TEXT (e.g. 'realised' "
                              "or 'forecast') instead of the best across all sections")
    parser.add_argument("--train-start", metavar="DATE",
                         help="with --best-vs-oracle, the training window's start, "
                              "for the title only -- the results file only records "
                              "what a checkpoint was scored on, not trained on")
    parser.add_argument("--train-end", metavar="DATE",
                         help="with --best-vs-oracle, the training window's end")
    args = parser.parse_args(argv)

    if bool(args.train_start) != bool(args.train_end):
        raise SystemExit("--train-start and --train-end must be given together")
    trainWindow = (args.train_start, args.train_end) if args.train_start else None

    sections = loadResultsFile(args.path)
    region = sections[0].region
    algo = sections[0].algo

    if args.show:
        import matplotlib.pyplot as plt
        if args.best_vs_oracle:
            buildBestSeedFigure(sections, region, algo, trainWindow=trainWindow,
                                 source=args.source)
        else:
            buildFigure(sections, region, algo)
        plt.show()
        return 0

    destination = args.out or args.path.with_suffix(".png")
    if args.best_vs_oracle:
        saveBestSeedFigure(sections, region, algo, destination, trainWindow=trainWindow,
                            source=args.source)
    else:
        saveFigure(sections, region, algo, destination)
    print(f"wrote          {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
