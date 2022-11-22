import random
from typing import Tuple

import cv2
import numpy as np

from augraphy.augmentations.lib import make_white_transparent
from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class ImageOverlay(Augmentation):
    """Takes a background and foreground image and overlays foreground somewhere
    on background. Not all of foreground will necessarily be visible; some may
    be cut off by the edge of the background image.

    :param foreground: the image to overlay on the background document
    :param position: a pair of x and y coordinates to place the foreground image
        If not given, the foreground will be randomly placed.
    :param p: the probability this augmentation will be applied
    """

    def __init__(self, foreground: np.ndarray, position: Tuple[int, int] = None, p: float = 1):
        self.foreground = foreground
        self.position = position
        super().__init__(p=p)

    def workspace(self, background: np.ndarray) -> np.ndarray:
        """Creates an empty image on which to do the overlay operation

        :param background: The background document image.
        """

        xdim = background.shape[0] + (2 * self.foreground.shape[0])
        ydim = background.shape[1] + (2 * self.foreground.shape[1])

        return cv2.cvtColor(
            np.ones((xdim, ydim, 3), dtype=np.uint8),
            cv2.COLOR_RGB2RGBA,
        )

    def layerForeground(self, ambient: np.ndarray, xloc: int, yloc: int) -> np.ndarray:
        """Put self.foreground at (xloc,yloc) on ambient

        :param ambient: The initial ambient image.
        :param xloc: Coordinate of x start location.
        :param yloc: Coordinate of y start location.
        """
        xstop = xloc + self.foreground.shape[0]
        ystop = yloc + self.foreground.shape[1]
        fg = cv2.cvtColor(np.uint8(self.foreground), cv2.COLOR_RGB2RGBA)
        bg = cv2.cvtColor(ambient, cv2.COLOR_RGB2RGBA)

        alpha_fg = fg[:, :, 3] / 255.0
        alpha_bg = 1 - alpha_fg
        for c in range(0, 3):
            ambient[xloc:xstop, yloc:ystop, c] = (alpha_bg * ambient[xloc:xstop, yloc:ystop, c]) + (
                alpha_fg * fg[:, :, c]
            )
        return ambient

    def overlay(self, background: np.ndarray) -> np.ndarray:
        """Centers the background image over workspace, then places foreground
        somewhere on the workspace, and finally crops to the
        background dimension

        :param background: Background image of overlaying process.
        """

        # Get the boundaries of the background image
        xstart = self.foreground.shape[0]
        ystart = self.foreground.shape[1]
        xstop = xstart + background.shape[0]
        ystop = ystart + background.shape[1]

        # Build the array we'll do work in
        ambient = self.workspace(background)

        # Center the background image
        ambient[xstart:xstop, ystart:ystop] = background

        if self.position is None:
            # Choose somewhere to put the foreground
            xloc = random.randrange(0, xstop)
            yloc = random.randrange(0, ystop)

        else:
            xloc = self.position[0] + xstart
            yloc = self.position[1] + ystart

        # Place the foreground at (xloc,yloc)
        ambient = self.layerForeground(ambient, xloc, yloc)

        # Crop the workspace to the original background image dimensions
        cropped = ambient[xstart:xstop, ystart:ystop]

        return cropped

    def __repr__(self):
        repstring = "ImageOverlay(\n" f"foreground={self.foreground},\n" f"position={self.position},\n" f"p={self.p})"

    def __call__(self, image: np.ndarray, force=False) -> np.ndarray:
        if force or self.should_run():
            image = image.copy
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            overlaid = self.overlay(image)
            return overlaid
