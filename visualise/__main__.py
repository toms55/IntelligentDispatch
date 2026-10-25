"""Plot a results/<REGION>.md file.

    uv run python -m visualise results/NSW1.md
    uv run python -m visualise results/NSW1.md --out NSW1.png
    uv run python -m visualise results/NSW1.md --show

    uv run python -m visualise results/NSW1.md --best-vs-oracle \\
        --train-start 2024-01-01T04:00 --train-end 2025-01-01T04:00

    uv run python -m visualise results/NSW1.md --compare-sources \\
        --train-start 2024-01-01T04:00 --train-end 2025-01-01T04:00

    uv run python -m visualise results/rolling2023_profit_all --region VIC1 \\
        --compare-sources --out VIC1.png
"""
import argparse
from pathlib import Path

from visualise.parse import loadResultsFile
from visualise.plots import (buildBestSeedFigure, buildFigure, buildSourceComparisonFigure,
                            saveBestSeedFigure, saveFigure, saveSourceComparisonFigure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path,
                         help="a results/<REGION>.md file, or a rolling backtest "
                              "results directory (then --region is needed)")
    parser.add_argument("--region", metavar="REGION",
                         help="with a rolling results directory, which region to read")
    parser.add_argument("--algo", metavar="NAME",
                         help="with a rolling results directory holding several algos, "
                              "which one to read")
    parser.add_argument("--train-mode", metavar="NAME",
                         help="with a rolling results directory holding several train "
                              "modes, which one to read")
    parser.add_argument("--out", type=Path, metavar="PATH",
                         help="where to save the figure; defaults to the input path "
                              "with a .png suffix")
    parser.add_argument("--show", action="store_true",
                         help="display the figure in a window instead of saving it")
    parser.add_argument("--best-vs-oracle", action="store_true",
                         help="plot just the single best seed against the oracle, "
                              "instead of the default three-panel overview")
    parser.add_argument("--compare-sources", action="store_true",
                         help="plot the best seed against the oracle once per price "
                              "source, side by side in one figure")
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

    if args.path.is_dir():
        if not args.region:
            raise SystemExit("--region is needed with a rolling results directory")
        from visualise.rolling import sectionsFromRolling, trainNote
        sections = sectionsFromRolling(args.path, args.region, algo=args.algo,
                                        trainMode=args.train_mode)
        # A rolling run retrains before every test week, so it has no one train window.
        trainWindow = trainWindow or trainNote(args.path, args.region)
    else:
        sections = loadResultsFile(args.path)
    region = sections[0].region
    algo = sections[0].algo

    if args.show:
        import matplotlib.pyplot as plt
        if args.compare_sources:
            buildSourceComparisonFigure(sections, region, algo, trainWindow=trainWindow)
        elif args.best_vs_oracle:
            buildBestSeedFigure(sections, region, algo, trainWindow=trainWindow,
                                 source=args.source)
        else:
            buildFigure(sections, region, algo)
        plt.show()
        return 0

    destination = args.out or (args.path / f"{args.region}.png" if args.path.is_dir()
                                else args.path.with_suffix(".png"))
    # `--out results/figures/x.png` names a directory the checkout may not hold, so
    # make it rather than fail inside savefig.
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.compare_sources:
        saveSourceComparisonFigure(sections, region, algo, destination,
                                    trainWindow=trainWindow)
    elif args.best_vs_oracle:
        saveBestSeedFigure(sections, region, algo, destination, trainWindow=trainWindow,
                            source=args.source)
    else:
        saveFigure(sections, region, algo, destination)
    print(f"wrote          {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
