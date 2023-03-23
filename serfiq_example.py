import cv2
import loguru

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


def main():
    model = SER_FIQ(gpu=0)
    test_imgs = ["./data/test_img.jpeg", "./data/test_img2.jpeg"]
    for test_img in test_imgs:
        score = get_image_score(model=model, image=test_img)
        log.info("SER-FIQ quality for {}: {:.3f}", test_img, score)


if __name__ == "__main__":
    main()
