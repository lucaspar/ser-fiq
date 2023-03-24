"""Example of how to use SER-FIQ model."""
import glob
import os
from typing import Dict, List

import cv2
import loguru
import polars as pl

from face_image_quality import SER_FIQ

log = loguru.logger


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


def get_scores_from_video(
    model: SER_FIQ, video: str, break_early: bool = False
) -> pl.DataFrame:
    """Runs SER-FIQ prediction on a video."""
    cap = cv2.VideoCapture(video)
    scores = dict()
    log.info("Processing video: {}", video)
    # TOOD: load ROI info for this video and crop the image before running the inference
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        aligned_img = model.apply_mtcnn(frame)

        if aligned_img is not None:
            score = model.get_score(aligned_img, T=100, alpha=130, r=0.88)
            if score is None:
                continue
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            scores[frame_number] = {
                "score": score,
                "frame_number": frame_number,
                "video": video,
            }
            if len(scores) % 100 == 0:
                log.info("Processed {} frames", len(scores))
            if break_early and len(scores) > 10:
                log.info("Breaking early")
                break

    df_scores = pl.from_dicts(list(scores.values()))
    return df_scores


def main():
    # read GPUs to use from CUDA_VISIBLE_DEVICES if set, otherwise use 0
    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    available_gpus = [int(g) for g in available_gpus.split(",")]
    model = SER_FIQ(gpu=available_gpus[0])

    # test_imgs = ["./data/test_img.jpeg", "./data/test_img2.jpeg"]
    # for test_img in test_imgs:
    #     score = get_image_score(model=model, image=test_img)
    #     log.info("SER-FIQ quality for {}: {:.3f}", test_img, score)

    # run it on a list of videos
    dir_videos = "./data/videos/"
    dir_score_outputs = "./data/scores/"
    log.info("Processing videos from {}", dir_videos)
    # videos = glob.glob(dir_videos + "**/*.mp4", recursive=True)

    videos = [
        "./data/videos/BGC2.0/BRS2/vol1/G02003/controlled/rand/wb/G02003_set1_rand_HBW8PR2_004084F4364B_6be70d9a.mp4",
        "./data/videos/BGC2.0/BRS2/vol1/G02003/controlled/rand/wb/G02003_set1_rand_HBW8PR2_004084F4366B_8d470c63.mp4",
        "./data/videos/BGC2.0/BRS2/vol1/G02003/controlled/rand/wb/G02003_set1_rand_HBW8PR2_004084F43676_a68be590.mp4",
    ]

    # loop over videos to get scores
    is_demo = True
    if is_demo:
        videos = videos[:1]  # TODO: remove cap
    for video in videos:
        csv_name = os.path.join(dir_score_outputs, video + ".csv")
        os.makedirs(os.path.dirname(csv_name), exist_ok=True)
        df_scores = get_scores_from_video(model=model, video=video, break_early=is_demo)
        df_scores.write_csv(csv_name)
        # save dataframe with video scores


if __name__ == "__main__":
    main()
