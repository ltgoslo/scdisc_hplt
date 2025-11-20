#! python3

import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from smart_open import open


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--data",
        "-d",
        help="Path to the data file (numbers and time periods, separated with whitespace)",
        required=True,
    )
    arg("--plot", "-p", help="Name of the file to save the plot")

    args = parser.parse_args()

    periods = []
    counts = []

    for line in open(args.data):
        instance = line.strip()
        number, period = instance.split()
        periods.append(period)
        counts.append(int(number))

    assert len(periods) == len(counts)

    # Sorting by time periods:
    indices = np.argsort(periods)
    periods = [periods[i] for i in indices]
    counts = [counts[i] for i in indices]

    fig, ax = plt.subplots()

    ax.tick_params("x", rotation=45)
    plt.rcParams["font.size"] = 10
    ax.tick_params(axis="both", labelsize=4)

    ax.bar(periods, counts)

    ax.set_ylabel("Document counts")
    ax.set_xlabel("Time periods (years or crawls)")
    ax.set_title(f"Document count distribution by time in {args.data.split("/")[0]}")

    if args.plot:
        plt.savefig(args.plot, dpi=300)
        logger.info(f"Plot saved to {args.plot}")
    else:
        plt.show()
