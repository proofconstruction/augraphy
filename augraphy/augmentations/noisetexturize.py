import random
from typing import Tuple

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class NoiseTexturize(Augmentation):
    """Creates a random noise based texture pattern to emulate paper textures.
    Consequently applies noise patterns to the original image from big to small.

    :param sigma_range: Defines bounds of noise fluctuations.
    :param turbulence_range: Defines how quickly big patterns will be
        replaced with the small ones. The lower value -
        the more iterations will be performed during texture generation.
    :param p: The probability this Augmentation will be applied.
    """

    def __init__(
        self,
        sigma_range: Tuple[int, int] = (3, 10),
        turbulence_range: Tuple[int, int] = (2, 5),
        p: float = 1,
    ):
        """Constructor method"""
        super().__init__(p=p)
        self.sigma_range = sigma_range
        self.turbulence_range = turbulence_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"NoiseTexturize(sigma_range={self.sigma_range}, turbulence_range={self.turbulence_range}, p={self.p})"

    def noise(self, width: int, height: int, channel: int, ratio: int, sigma: int) -> np.ndarray:
        """The function generates an image, filled with gaussian nose. If ratio
        parameter is specified, noise will be generated for a lesser image and
        then it will be upscaled to the original size. In that case noise will
        generate larger square patterns. To avoid multiple lines, the upscale
        uses interpolation.

        :param width: Width of generated image.
        :param height: Height of generated image.
        :param channel: Channel number of generated image.
        :param ratio: The size of generated noise "pixels".
        :param sigma: Defines bounds of noise fluctuations.
        """
        mean = 0
        # assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
        # assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

        h = int(height / ratio)
        w = int(width / ratio)

        if h == 0:
            h = 1
        if w == 0:
            w = 1

        gaussian = np.vectorize(lambda x: random.gauss(mean, sigma))

        result = gaussian(np.array((w, h)))

        result = cv2.resize(
            result,
            dsize=(width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        # for multiple channels input, convert result to multiple channels
        if channel:
            result = np.stack([result, result, result], axis=2)

        return result

    # Applies the Augmentation to input data.
    def __call__(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        if force or self.should_run():
            image = image.copy()

            sigma = random.randint(self.sigma_range[0], self.sigma_range[1])
            turbulence = random.randint(
                self.turbulence_range[0],
                self.turbulence_range[1],
            )

            result = image.astype(float)
            rows, cols = image.shape[:2]
            if len(image.shape) > 2:
                channel = image.shape[2]
            else:
                channel = 0

            ratio = cols
            while not ratio == 1:
                result += self.noise(cols, rows, channel, ratio, sigma=sigma)
                ratio = (ratio // turbulence) or 1
            cut = np.clip(result, 0, 255)

            cut = cut.astype(np.uint8)
            return cut
