import random
from typing import Tuple

import numpy as np

from augraphy.augmentations.lowinkline import LowInkLine


class LowInkPeriodicLines(LowInkLine):
    """Creates a set of lines that repeat in a periodic fashion throughout the
    image.

    :param count_range: Pair of ints determining the range from which to sample
           the number of lines to apply.
    :param period_range: Pair of ints determining the range from which to sample
           the distance between lines.
    :param use_consistent_lines: Whether or not to vary the width and alpha of
           generated low ink lines.
    :param noise_probability: The probability to add noise into the generated lines.
    :param p: The probability that this Augmentation will be applied.
    """

    def __init__(
        self,
        count_range: Tuple[int, int] = (2, 5),
        period_range: Tuple[int, int] = (10, 30),
        use_consistent_lines: bool = True,
        noise_probability: float = 0.1,
        p: float = 1,
    ):
        """Constructor method"""
        super().__init__(
            use_consistent_lines=use_consistent_lines,
            noise_probability=noise_probability,
            p=p,
        )
        self.count_range = count_range
        self.period_range = period_range

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"LowInkPeriodicLines(count_range={self.count_range}, period_range={self.period_range}, use_consistent_lines={self.use_consistent_lines}, p={self.p})"

    def add_periodic_transparency_line(self, mask: np.ndarray, line_count: int, offset: int, alpha: int) -> None:
        """Creates horizontal lines of some opacity over the input image, at y-positions determined by the offset and line_count.

        :param mask: The image to apply the line to.
        :param line_count: The number of lines to generate.
        :param offset: How far from the edge of the image to generate lines.
        :param alpha: The opacity of the lines.
        """
        period = mask.shape[0] // line_count

        for y in range(mask.shape[0] - offset):
            if (period != 0) and (
                y % period == 0
            ):  # period can't be zero here, else there would be zero division error
                self.add_transparency_line(mask, y + offset, alpha)

    def add_periodic_transparency_lines(self, mask: np.ndarray, lines: int, line_periods: int) -> None:
        """Creates horizontal lines of random opacity over the input image, at
        random intervals.

        :param mask: The image to apply the line to.
        :param lines: How many lines to add to the image.
        :param line_periods: The distance between lines.
        """
        period = mask.shape[0] // line_periods
        self.add_periodic_transparency_line(
            mask,
            line_periods,
            offset=random.randint(0, 5),
            alpha=random.randint(96, 255),
        )

        for i in range(lines):
            self.add_periodic_transparency_line(
                mask,
                line_periods,
                offset=random.randint(0, period),
                alpha=random.randint(16, 96),
            )

    # Applies the Augmentation to input data.
    def __call__(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        if force or self.should_run():
            image = image.copy()
            count = random.randint(self.count_range[0], self.count_range[1])
            period = random.randint(self.period_range[0], self.period_range[1])

            for i in range(count):
                self.add_periodic_transparency_lines(image, count, period)

            return image
