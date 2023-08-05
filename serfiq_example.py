"""Example of how to use SER-FIQ model."""
import glob
import os
from typing import Optional

import cv2
import loguru
import polars as pl
from polars.dataframe.groupby import GroupBy
from tqdm.auto import tqdm

from face_image_quality import SER_FIQ

log = loguru.logger

# silence part of MXNET logging
os.environ["MXNET_SUBGRAPH_VERBOSE"] = "0"


def get_image_score(model: SER_FIQ, image: str) -> float:
    """Runs SER-FIQ prediction on a single image."""
    loaded_image = cv2.imread(image)
    aligned_img = model.apply_mtcnn(loaded_image)

    assert aligned_img is not None, "No face detected in image"

    # T=100 (default) is a good choice
    # alpha and r parameters can be used to scale the score distribution
    score = model.get_score(aligned_img, T=100, alpha=130, r=0.88)
    assert score is not None, "No face detected in image"

    return score


def prepare_tracklets(
    csv_tracklets: str, n_rows: Optional[int] = None
) -> GroupBy[pl.DataFrame]:
    """Loads tracklets from disk and group them by video."""
    # load from disk
    log.info(f"Loading tracklets from {csv_tracklets}")
    df_tracklets = pl.read_csv(csv_tracklets, n_rows=n_rows)
    log.info(f"Loaded {len(df_tracklets):,} tracklets")
    log.info(f"Columns: {df_tracklets.columns}")

    # drop columns with empty names
    drops = [c for c in df_tracklets.columns if c == ""]
    df_tracklets = df_tracklets.drop(drops)

    # group by video
    df_tracklet_groups = df_tracklets.groupby("source_path")

    return df_tracklet_groups


def get_scores_from_video(
    model: SER_FIQ, video_path: str, df_rois: pl.DataFrame, break_early: bool = False
) -> Optional[pl.DataFrame]:
    """Runs SER-FIQ prediction on a video."""
    scores = dict()
    log.info(f"Processing video: {video_path} using {df_rois.height} ROIs")
    # sort df_rois by frame_number
    df_rois = df_rois.sort("frame_number")
    prog_rois = tqdm(
        df_rois.iter_rows(named=True),
        total=df_rois.height,
        ncols=70,
        colour="blue",
        unit="ROI",
    )
    cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    try:
        # fetching first ROI and frame
        frame_number = -1
        for roi in prog_rois:
            # fetching next ROI
            original_frame = None

            # read frames until we reach the ROI's frame number
            while frame_number < roi["frame_number"]:
                if not cap.isOpened():
                    log.warning(f"Prematurely reached end of video {video_path}")
                    break
                ret, original_frame = cap.read()
                if not ret:
                    # log.debug("Reached end of video")
                    break
                # if frame_number is different from ROI's frame no., go to the next one
                frame_number = cap.get(
                    cv2.CAP_PROP_POS_FRAMES  # pylint: disable=no-member
                )
                # this is ideally '==', but to avoid endless loops entirely, we use '>='
                if frame_number >= roi["frame_number"]:
                    break
                # if difference between frame_number and ROI's frame no. is too large,
                # use cap.set() to jump to the ROI's frame number
                is_too_far = frame_number + 10 < roi["frame_number"]
                if is_too_far:
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,  # pylint: disable=no-member
                        roi["frame_number"] - 1,
                    )

            # sanity checks
            if original_frame is None or original_frame.size == 0:
                log.warning(
                    f"Frame {frame_number} is empty, skipping ROI {roi['roi_id']}"
                )
                continue
            if frame_number != roi["frame_number"]:
                log.warning(
                    f"Skipping ROI {roi['roi_id']}, as frame number {frame_number} does "
                    + f"not match ROI's frame number {roi['frame_number']}"
                )
                continue
            assert roi["x1"] < roi["x2"], f"Unexpected x1 >= x2 for ROI {roi['roi_id']}"
            assert roi["y1"] < roi["y2"], f"Unexpected y1 >= y2 for ROI {roi['roi_id']}"

            # crop frame to roi coords
            cropped_frame = original_frame[roi["y1"] : roi["y2"], roi["x1"] : roi["x2"]]
            aligned_img = model.apply_mtcnn(cropped_frame)

            if aligned_img is not None:
                score = model.get_score(aligned_img, T=100, alpha=130, r=0.88)
                if score is None:
                    continue
                roi_shape_str = (
                    "(" + ", ".join([str(c) for c in cropped_frame.shape]) + ")"
                )
                scores[frame_number] = {
                    "serfiq_score": score,
                    "frame_number": frame_number,
                    "roi_shape": roi_shape_str,
                    "video_path": video_path,
                    **roi,
                }
                if len(scores) % 100 == 0:
                    log.info("Processed {} frames", len(scores))
                if break_early and len(scores) > 2:
                    log.info("Breaking early")
                    break
    finally:
        cap.release()

    if len(scores) == 0:
        log.warning(f"No scores for video {video_path}")
        return None
    df_scores = pl.from_dicts(list(scores.values()))
    return df_scores


def main():
    """Entrypoint for processing videos."""
    # read GPUs to use from CUDA_VISIBLE_DEVICES if set, otherwise use the CPU
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if available_gpus is not None:
        available_gpus = [int(g) for g in available_gpus.split(",")]

    # test_imgs = ["./data/test_img.jpeg", "./data/test_img2.jpeg"]
    # for test_img in test_imgs:
    #     score = get_image_score(model=model, image=test_img)
    #     log.info("SER-FIQ quality for {}: {:.3f}", test_img, score)

    # paths
    dir_videos = "./data/videos/"
    dir_meta = "./data/meta/"
    dir_score_outputs = "./data/scores/"
    log.info("Processing videos from {}", dir_videos)
    csv_tracklets = os.path.join(dir_meta, "all/BRS/tracklets_td32_2023-03-01.csv")

    # make sure all dirs exist
    for dir_name in [dir_videos, dir_meta, dir_score_outputs]:
        assert os.path.exists(dir_name), f"Directory {dir_name} does not exist"

    is_demo = available_gpus is None  # always run in demo mode if using CPU
    if is_demo:
        log.warning(">>> Running in demo mode <<<")
    df_tracklets = prepare_tracklets(
        csv_tracklets=csv_tracklets, n_rows=10_000 if is_demo else None
    )
    log.info(df_tracklets.head())

    # load model
    model = SER_FIQ(gpu=available_gpus[0] if available_gpus is not None else None)

    for source_path, df_tracklet in df_tracklets:
        # get the first video_id to save scores
        video_id = df_tracklet[0, ["video_id"]].item()
        matches = os.path.join(dir_videos, "**", source_path)  # type: ignore
        sources = glob.glob(matches)
        if len(sources) == 0:
            log.warning(f"Could not find video matching {matches}")
            continue
        source_path = sources[0]
        csv_name = os.path.join(dir_score_outputs, f"{video_id}.csv")
        # if csv exists, skip to next video
        if os.path.exists(csv_name):
            continue
        else:
            # create empty file to avoid processing the same video twice
            with open(csv_name, "w", encoding="utf-8") as csv_ptr:
                csv_ptr.write("")
        log.info(f"Processing video path '{source_path}'")
        log.info(f"Saving video scores to '{csv_name}'")
        os.makedirs(os.path.dirname(csv_name), exist_ok=True)
        df_scores = get_scores_from_video(
            model=model,
            video_path=source_path,
            df_rois=df_tracklet,
            break_early=is_demo,
        )
        if df_scores is None:
            continue
        if is_demo:
            log.info(df_scores.head())
        df_scores.write_csv(csv_name)
        if is_demo:
            break


if __name__ == "__main__":
    main()
