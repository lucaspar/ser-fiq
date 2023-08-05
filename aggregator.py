"""Aggregates all generated scores into a single file and draw some plots."""
import glob
import os
from typing import Any, Dict, List, Union

import cv2
import loguru
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from polars import exceptions
from scipy.stats import norm

log = loguru.logger


def load_all_scores(dir_scores: str) -> pl.DataFrame:
    """Loads all CSV score tables into a single DataFrame."""
    # get all csv names matching a uuid-like pattern
    matcher = os.path.join(dir_scores, "*-*-*-*-*.csv")
    csv_scores = glob.glob(matcher)
    log.info(f"Discovered {len(csv_scores):,} score files")
    dfs = list()
    for csv_score in csv_scores:
        try:
            df_score = pl.read_csv(csv_score)
        except exceptions.NoDataError:
            continue
        dfs.append(df_score)
    df_scores = pl.concat(dfs)
    return df_scores


def plot_scores(
    df_scores: Union[pl.DataFrame, pd.DataFrame],
    dir_plots: str = "plots",
    display: bool = True,
) -> None:
    """Plots scores distribution."""
    if isinstance(df_scores, pl.DataFrame):
        df_scores = df_scores.to_pandas()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = iter(axes.flatten()) if hasattr(axes, "flatten") else iter([axes])  # type: ignore
    min_quality_threshold: float = 0.1

    axis = next(axes)
    assert axis is not None, "No axis found"
    axis = df_scores["serfiq_score"].hist(ax=axis, bins=100, histtype="step")  # type: ignore
    axis.set_title("SER-FIQ score distribution")  # type: ignore
    axis.set_xlabel("SER-FIQ score")  # type: ignore
    axis.set_ylabel("Count")  # type: ignore
    axis.axvline(min_quality_threshold, color="red", linestyle="--")  # type: ignore
    legend = ["SER-FIQ score", f"Quality threshold ({min_quality_threshold})"]
    axis.legend(legend)  # type: ignore
    axis.grid(True)  # type: ignore

    axis = next(axes)
    assert axis is not None, "No axis found"
    axis = df_scores["serfiq_score"].hist(ax=axis, bins=100, histtype="step", log=True)  # type: ignore
    axis.set_title("SER-FIQ score distribution (log scale)")  # type: ignore
    axis.set_xlabel("SER-FIQ score")  # type: ignore
    axis.set_ylabel("Count")  # type: ignore
    axis.axvline(min_quality_threshold, color="red", linestyle="--")  # type: ignore
    axis.legend(legend)  # type: ignore
    axis.grid(True)  # type: ignore

    # group by tracklet_id and get the mean and std
    # log.info(df_scores.columns)
    # df_scores = df_scores[df_scores["serfiq_score"] > min_quality_threshold]
    df_tracklets_agg = df_scores.groupby("tracklet_id").agg(
        tracklet_mean=pd.NamedAgg(column="serfiq_score", aggfunc=lambda x: x.mean()),
        tracklet_std=pd.NamedAgg(column="serfiq_score", aggfunc=lambda x: x.std()),
    )
    df_tracklets_agg = df_tracklets_agg[
        df_tracklets_agg["tracklet_mean"] > min_quality_threshold
    ]
    # scatter plot of mean and std
    axis = next(axes)
    assert axis is not None, "No axis found"
    axis.scatter(  # type: ignore
        df_tracklets_agg["tracklet_mean"],
        df_tracklets_agg["tracklet_std"],
        s=1,
    )
    title = f"SER-FIQ scores agg. by tracklet | Mean (>{min_quality_threshold}) vs. Std"
    xlabel = "SER-FIQ mean for tracklet"
    ylabel = "SER-FIQ standard deviation for tracklet"
    axis.set_title(title)  # type: ignore
    axis.set_xlabel(xlabel)  # type: ignore
    axis.set_ylabel(ylabel)  # type: ignore
    axis.grid(True)  # type: ignore

    # same plot with 2D hexbin
    axis = next(axes)
    assert axis is not None, "No axis found"
    hex_bin = axis.hexbin(  # type: ignore
        df_tracklets_agg["tracklet_mean"],
        df_tracklets_agg["tracklet_std"],
        gridsize=7,
        mincnt=1,
        cmap="viridis",
    )
    cb = fig.colorbar(hex_bin, ax=axis)
    cb.set_label("Count")
    axis.set_title(title)  # type: ignore
    axis.set_xlabel(xlabel)  # type: ignore
    axis.set_ylabel(ylabel)  # type: ignore
    # show colorbar
    # fig.colorbar(plt.cm.ScalarMappable(), ax=axis)  # type: ignore
    axis.grid(True)  # type: ignore

    # save plot
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig_name = os.path.join(dir_plots, f"all_scores.{ext}")
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        fig.savefig(fig_name)
    if display:
        plt.show()


def plot_samples(
    samples: List[Dict[str, Any]], display: bool = True, dir_plots: str = "plots"
) -> None:
    """Displays a grid of sampled faces with their SER-FIQ scores."""
    num_samples = len(samples)
    n_rows = int(num_samples**0.5)
    n_cols = int(num_samples / n_rows)
    fig, axes_original = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = (
        iter(axes_original.flatten())  # type: ignore
        if hasattr(axes_original, "flatten")
        else iter([axes_original])
    )
    for axis in axes:
        assert axis is not None, "No axis found"
        axis.axis("off")  # type: ignore
    axes = (
        iter(axes_original.flatten())  # type: ignore
        if hasattr(axes_original, "flatten")
        else iter([axes_original])
    )
    for sample in samples:
        axis = next(axes)
        assert axis is not None, "No axis found"
        axis.imshow(sample["face"])  # type: ignore
        axis.set_title(f"Score: {sample['score']:.3f}")  # type: ignore

    fig.suptitle("Sampled faces with their SER-FIQ scores", fontsize=16)
    fig.tight_layout()
    fig_name = os.path.join(dir_plots, f"sampled_faces_{num_samples}.png")
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    fig.savefig(fig_name)
    if display:
        plt.show()


def load_samples_from_disk(
    df_sample: pl.DataFrame,
    dir_videos: str = "data/videos",
) -> List[Dict[str, Any]]:
    """Loads samples from disk, converting to RGB and cropping the face."""
    samples: list[dict[str, Any]] = list()
    for row in df_sample.iter_rows(named=True):
        # columns: serfiq_score,frame_number,roi_shape,video_path,roi_id,x1,y1,x2,y2,
        #   frame_id,subject_id,db_id,folder,video_id,source_path,type,tracklet_id
        matcher = os.path.join(dir_videos, "*", row["source_path"])
        matches = glob.glob(matcher, recursive=True)
        if len(matches) == 0:
            log.warning(f"No video found for {matcher}")
            continue
        video_path = matches[0]
        log.info(f"Opening video {video_path}")
        cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
        if not cap.isOpened():
            log.warning(f"Could not open video {video_path}")
            continue
        try:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,  # pylint: disable=no-member
                row["frame_number"],
            )
            ret, frame = cap.read()
            if not ret:
                log.warning(
                    f"Could not read frame {row['frame_number']} from {video_path}"
                )
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
            roi_x1, roi_y1, roi_x2, roi_y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            face = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            samples.append(
                {
                    "score": row["serfiq_score"],
                    "face": face,
                }
            )
        finally:
            cap.release()

    return samples


def face_sampler(
    df_scores: pl.DataFrame,
    dir_videos: str = "data/videos",
    dir_plots: str = "plots",
    num_samples: int = 9,
    display: bool = True,
) -> None:
    """Samples faces with their scores."""

    # randomly sample faces
    df_sample = df_scores.sample(num_samples, seed=42)
    df_sample = df_sample.sort("serfiq_score")

    # load them from videos on disk
    samples = load_samples_from_disk(df_sample=df_sample, dir_videos=dir_videos)

    # display samples in a grid
    plot_samples(
        samples=samples,
        display=display,
        dir_plots=dir_plots,
    )


def compute_serfiq_difficulty_scores(df_scores: pl.DataFrame) -> pl.DataFrame:
    """Computes SER-FIQ difficulty scores."""

    # get mean and stdev of scores
    mean = df_scores["serfiq_score"].mean()
    std = df_scores["serfiq_score"].std()

    def fn_difficulty(ser_fiq_score: float) -> float:
        """Passes the raw score through a cumulative distribution function.

        In order to standardize it and make it a difficulty score.
        """
        return 1 - norm.cdf(ser_fiq_score, loc=mean, scale=std)

    df_diff = df_scores.with_columns(
        [
            pl.col(
                "serfiq_score",
            )
            .apply(fn_difficulty)
            .alias("difficulty"),
        ]
    )
    return df_diff


def plot_difficulties_and_scores(
    df_scores: pl.DataFrame,
    display: bool = True,
    dir_plots: str = "plots",
) -> None:
    """Plots the SER-FIQ scores and difficulty scores."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].hist(df_scores["serfiq_score"], bins=100, histtype="step")
    axes[0].set_title("SER-FIQ scores")
    axes[1].hist(df_scores["difficulty"], bins=100, histtype="step")
    axes[1].set_title("Difficulty scores")
    fig.tight_layout()
    fig_name = os.path.join(dir_plots, "difficulties_and_scores.png")
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    fig.savefig(fig_name)
    if display:
        plt.show()


def main():
    """Main function to aggregate all scores into a single CSV file."""
    # parameters
    dir_scores = "data/scores"
    dir_plots = "plots"

    # load scores and aggregate them
    df_scores = None
    aggregate_files = False
    if aggregate_files:
        df_scores = load_all_scores(dir_scores)
        log.info(f"Loaded {len(df_scores):,} scores")
        log.info(df_scores.head(5))
        df_scores.write_csv(f"data/scores/agg_scores_{len(df_scores):,}.csv")
    else:
        log.warning("Not aggregating scores")

    # compute ser-fiq difficulty scores
    if df_scores is None:
        agg_file_matcher = os.path.join(dir_scores, "agg_*.csv")
        agg_file = glob.glob(agg_file_matcher)
        num_rows = 100_000
        df_scores = pl.read_csv(agg_file[0], n_rows=num_rows)
        log.info(f"Loaded {len(df_scores):,} scores")

    df_scores = compute_serfiq_difficulty_scores(df_scores=df_scores)
    plot_difficulties_and_scores(df_scores=df_scores, dir_plots=dir_plots, display=True)

    # plots and previews
    # plot_scores(df_scores=df_scores, dir_plots=dir_plots, display=True)
    # face_sampler(
    #     df_scores=df_scores,
    #     dir_videos=dir_videos,
    #     dir_plots=dir_plots,
    #     num_samples=num_samples,
    #     display=True,
    # )


if __name__ == "__main__":
    main()
