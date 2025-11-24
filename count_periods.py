#! python3

import argparse
import pandas as pd
import os
import logging

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
        help="Path to the directory with language stats subdirectories",
        default="languages/",
    )
    arg(
        "--outf",
        "-o",
        help="Path to the output dataframe with all the stats aggregated",
        default="dia_lang_stats.tsv",
    )
    arg(
        "--threshold",
        "-t",
        type=int,
        help="How many documents we want per period?",
        default=1000000,
    )
    args = parser.parse_args()

    counts = {}

    with os.scandir(args.data) as it:
        counter = 0
        for entry in it:
            if not entry.name.startswith(".") and entry.is_dir():
                logger.info(entry.name)
                counts[counter] = {}
                counts[counter]["Language"] = entry.name
                for line in open(f"{entry.name}/ts_sorted.txt"):
                    instance = line.strip()
                    number, year = instance.split()
                    counts[counter][year] = int(number)
                counter += 1

    df = pd.DataFrame.from_dict(counts, orient="index")

    periods = [["2011", "2012", "2013", "2014", "2015"], ["2020", "2021"], ["2024"]]
    periods_names = ["Ancient", "Covid", "Modern"]

    for name, years in zip(periods_names, periods):
        df[name] = df[years].sum(axis=1)

    df.to_csv(args.outf, sep="\t", index=False)

    df_periods = df[["Language"] + periods_names]
    logger.info("All our languages by period:")
    print(df_periods)

    missing = df_periods.loc[
        (df_periods[periods_names[0]] < args.threshold)
        | (df_periods[periods_names[1]] < args.threshold)
        | (df_periods[periods_names[2]] < args.threshold)
    ]
    logger.info(
        f"All languages with less than {args.threshold} documents in at least one period:"
    )
    print(missing)
    logger.info(f"{len(missing)} total problematic languages")
