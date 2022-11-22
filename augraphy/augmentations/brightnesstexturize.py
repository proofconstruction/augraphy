import random
from typing import Tuple
from typing import Union

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class BrightnessTexturize(Augmentation):
    """Creates a random noise in the brightness channel to emulate paper
    textures.

    :param texturize_range: Pair of floats determining the range from which to sample values
           for the brightness matrix. Suggested value = <1.
    :param deviation: Additional variation for the uniform sample.
    :param p: The probability that this Augmentation will be applied.
    """

    def __init__(self, texturize_range: Tuple[float, float] = (0.9, 0.99), deviation: float = 0.03, p: float = 1):
        """Constructor method"""
        super().__init__(p=p)
        self.low = texturize_range[0]
        self.high = texturize_range[1]
        self.deviation = deviation
        self.texturize_range = texturize_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BrightnessTexturize(texturize_range={self.texturize_range}, deviation={self.deviation}, p={self.p})"

    def compute_texture(self, hsv: np.ndarray) -> np.ndarray:
        # compute random value
        value = random.uniform(self.low, self.high)
        # convert to float (range 0-1)
        hsv = np.array(hsv, dtype=np.float64)

        # add noise using deviation
        low_value = value - (value * self.deviation)  # *random.uniform(0, deviation)
        max_value = value + (value * self.deviation)

        # apply noise
        makerand = np.vectorize(lambda x: random.uniform(low_value, max_value))
        brightness_matrix = makerand(np.zeros((hsv.shape[0], hsv.shape[1])))
        hsv[:, :, 1] *= brightness_matrix
        hsv[:, :, 2] *= brightness_matrix
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

        # convert back to uint8, apply bitwise not and convert to hsv again
        hsv = np.array(hsv, dtype=np.uint8)
        hsv = np.invert(hsv)
        hsv = np.array(hsv, dtype=np.float64)

        # add noise using deviation again
        new_low_value = value - (value * self.deviation)
        new_max_value = value + (value * self.deviation)

        # apply noise again
        makerand = np.vectorize(lambda x: random.uniform(new_low_value, new_max_value))
        brightness_matrix = makerand(np.zeros((hsv.shape[0], hsv.shape[1])))
        hsv[:, :, 1] *= brightness_matrix
        hsv[:, :, 2] *= brightness_matrix
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

        # convert back to uint8, apply bitwise not
        hsv = np.array(hsv, dtype=np.uint8)
        hsv = np.invert(hsv)

        return hsv

    # Applies the Augmentation to input data.
    def __call__(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        if force or self.should_run():
            image_output = image.copy()

            # for colour image
            if len(image.shape) > 2:
                hsv = cv2.cvtColor(image_output.astype("uint8"), cv2.COLOR_BGR2HSV)
            # for gray image
            else:
                bgr = hsv = cv2.cvtColor(
                    image_output.astype("uint8"),
                    cv2.COLOR_GRAY2BGR,
                )
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            hsv = self.compute_texture(hsv)
            image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # convert back to gray
            if len(image.shape) < 3:
                image_output = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

            return image_output
