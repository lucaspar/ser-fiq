from typing import List, Optional, Union

import cv2
import numpy as np
from skimage import transform as trans


def parse_lst_line(line: str):
    """Parses a line in .lst file."""
    vec = line.strip().split("\t")
    assert len(vec) >= 3
    aligned = int(vec[0])
    image_path = vec[1]
    label = int(vec[2])
    bbox = None
    landmark = None
    # print(vec)
    if len(vec) > 3:
        bbox = np.zeros((4,), dtype=np.int32)
        for i in xrange(3, 7):
            bbox[i - 3] = int(vec[i])
        landmark = None
        if len(vec) > 7:
            _l = []
            for i in xrange(7, 17):
                _l.append(float(vec[i]))
            landmark = np.array(_l).reshape((2, 5)).T
    # print(aligned)
    return image_path, label, bbox, landmark, aligned


def read_image(img_path: str, **kwargs) -> np.ndarray:
    """Reads image from file."""
    mode = kwargs.get("mode", "rgb")
    layout = kwargs.get("layout", "HWC")
    if mode == "gray":
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
        if mode == "rgb":
            # print('to rgb')
            img = img[..., ::-1]
        if layout == "CHW":
            img = np.transpose(img, (2, 0, 1))
    return img


def preprocess(
    original_img: Union[str, np.ndarray],
    bbox: Optional[Union[List[int], np.ndarray]] = None,
    landmark: Optional[np.ndarray] = None,
    **kwargs
):
    """Pre-processes image for face recognition."""
    if isinstance(original_img, str):
        img = read_image(original_img, **kwargs)
    else:
        img = original_img
    transformation_matrix = None
    image_size = []
    str_image_size = kwargs.get("image_size", "")
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(",")]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        sim_transform = trans.SimilarityTransform()
        sim_transform.estimate(dst, src)
        assert sim_transform.params is not None, "tform.params is None"
        transformation_matrix = sim_transform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if transformation_matrix is None:
        if bbox is None:
            # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 1 / 16)
            det[1] = int(img.shape[0] * 1 / 16)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            # use bounding box
            det = bbox
        margin = kwargs.get("margin", 44)
        limited_bbox = np.zeros(4, dtype=np.int32)
        limited_bbox[0] = int(np.maximum(det[0] - margin / 2, 0))
        limited_bbox[1] = int(np.maximum(det[1] - margin / 2, 0))
        limited_bbox[2] = int(np.minimum(det[2] + margin / 2, img.shape[1]))
        limited_bbox[3] = int(np.minimum(det[3] + margin / 2, img.shape[0]))
        cropped = img[
            limited_bbox[1] : limited_bbox[3], limited_bbox[0] : limited_bbox[2], :
        ]
        resized = (
            cv2.resize(cropped, (image_size[1], image_size[0]))
            if len(image_size) > 0
            else cropped
        )

        return resized
    else:  # do align using landmarks
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(M)
        warped = cv2.warpAffine(
            img,
            transformation_matrix,
            (image_size[1], image_size[0]),
            borderValue=0.0,
        )

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return warped
