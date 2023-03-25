"""Aggregates all generated scores into a single file and draw some plots."""
import glob
import os
from typing import Any, Dict, List, Union

import cv2
import loguru
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from polars import exceptions

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
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes = iter(axes.flatten()) if hasattr(axes, "flatten") else iter([axes])  # type: ignore

    axis = next(axes)
    assert axis is not None, "No axis found"
    axis = df_scores["serfiq_score"].hist(ax=axis, bins=100, histtype="step")  # type: ignore
    axis.set_title("SER-FIQ score distribution")  # type: ignore
    axis.set_xlabel("SER-FIQ score")  # type: ignore
    axis.set_ylabel("Count")  # type: ignore
    axis.grid(True)  # type: ignore
    fig.tight_layout()

    # save plot
    fig_name = os.path.join(dir_plots, "all_scores.png")
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


def main():
    """Main function to aggregate all scores into a single CSV file."""
    # parameters
    num_samples = 25
    dir_scores = "data/scores"
    dir_videos = "data/videos"
    dir_plots = "plots"

    # load scores and aggregate them
    df_scores = load_all_scores(dir_scores)
    log.info(f"Loaded {len(df_scores):,} scores")
    log.info(df_scores.head(5))
    df_scores.write_csv(f"data/scores/agg_scores_{len(df_scores):,}.csv")

    # plots and previews
    plot_scores(df_scores=df_scores, dir_plots=dir_plots, display=True)
    face_sampler(
        df_scores=df_scores,
        dir_videos=dir_videos,
        dir_plots=dir_plots,
        num_samples=num_samples,
        display=True,
    )


if __name__ == "__main__":
    main()
