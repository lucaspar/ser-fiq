"""
Authors: Jan Niklas Kolf, Philipp Terhörst

This code is licensed under the terms of the
    Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
    https://creativecommons.org/licenses/by-nc-sa/4.0/


Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from typing import Optional

import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from insightface.src import face_preprocess, mtcnn_detector


class SER_FIQ:
    def __init__(
        self,
        gpu: Optional[int] = 0,  # Which gpu should be used -> gpu id
        det: int = 0,  # Mtcnn option, 1= Use R+O, 0=Detect from beginning
    ):
        """
        Reimplementing Insightface's FaceModel class.
        Now the dropout output and the network output are returned after a forward pass.

        Parameters
        ----------
        gpu : int, optional
            The GPU to be used by Mxnet. The default is 0.
            If set to None, CPU is used instead.
        det : int, optional
            Mtcnn option, 1= Use R+0, 0= Detect from beginning. The default is 0.

        Returns
        -------
        None.

        """

        if gpu is None:
            self.device = mx.cpu()
        else:
            self.device = mx.gpu(gpu)

        self.insightface = gluon.nn.SymbolBlock.imports(
            "./insightface/model/insightface-symbol.json",
            ["data"],
            "./insightface/model/insightface-0000.params",
            ctx=self.device,
        )

        self.det_minsize = 50
        self.det_thresholds = [0.6, 0.7, 0.8]
        self.det = det

        self.preprocess = face_preprocess.preprocess

        thresholds = self.det_thresholds if det == 0 else [0.0, 0.0, 0.2]

        self.detector = mtcnn_detector.MtcnnDetector(
            model_folder="./insightface/mtcnn-model/",
            ctx=self.device,
            num_worker=1,
            accurate_landmark=True,
            threshold=thresholds,
        )

    def apply_mtcnn(self, face_image: np.ndarray):
        """
        Applies MTCNN Detector on the given face image and returns
        the cropped image.

        If no face could be detected None is returned.

        Parameters
        ----------
        face_image : np.ndarray
            Face imaged loaded via OpenCV.

        Returns
        -------
        Face Image : np.ndarray, shape (3,112,112).
        None, if no face could be detected

        """
        detected = self.detector.detect_face(face_image, det_type=self.det)

        if detected is None:
            return None

        bbox, points = detected

        if bbox.shape[0] == 0:
            return None

        points = points[0, :].reshape((2, 5)).T

        image = self.preprocess(face_image, bbox, points, image_size="112,112")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return np.transpose(image, (2, 0, 1))

    def get_score(
        self,
        aligned_img: np.ndarray,
        T: int = 100,
        alpha: float = 130.0,
        r: float = 0.88,
    ):
        """
        Calculates the SER-FIQ score for a given aligned image using T passes.

        Parameters
        ----------
        aligned_img : np.ndarray, shape (3, h, w)
            Aligned face image, in RGB format.
        T : int, optional
            Amount of forward passes to use. The default is 100.
        alpha : float, optional
            Stretching factor, can be chosen to scale the score values
        r : float, optional
            Score displacement

        Returns
        -------
        SER-FIQ score : float.

        """
        # if color channel is not the first dimension, swap dims.
        if aligned_img.shape[0] != 3:
            aligned_img = np.transpose(aligned_img, (2, 0, 1))

        # data validation
        assert len(aligned_img.shape) == 3, "Image must be in RGB format"
        assert aligned_img.shape[0] == 3, "Image must be in RGB format"

        # build the input of T forward passes
        input_blob = np.expand_dims(aligned_img, axis=0)
        repeated = np.repeat(input_blob, T, axis=0)
        gpu_repeated = mx.nd.array(repeated, ctx=self.device)

        output_tensor = self.insightface(gpu_repeated).asnumpy()

        norm = normalize(output_tensor, axis=1)

        # only get the upper triangle of the distance matrix
        euclidean_dist = euclidean_distances(X=norm, Y=norm)[np.triu_indices(T, k=1)]

        # calculate score as given in the paper
        score = 2 * (1 / (1 + np.exp(np.mean(euclidean_dist))))

        # normalize value based on alpha and r
        return 1 / (1 + np.exp(-(alpha * (score - r))))
