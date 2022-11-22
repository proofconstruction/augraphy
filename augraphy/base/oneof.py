import random
from typing import List
from typing import Tuple

import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationsequence import AugmentationSequence


class OneOf(Augmentation):
    """Given a list of Augmentations, selects one to apply.

    :param augmentations: A list of Augmentations to choose from.
    :param p: The probability that this augmentation will be applied.
    """

    def __init__(self, augmentations: List[Augmentation], p: float = 1):
        """Constructor method"""
        self.augmentations = augmentations
        self.augmentation_probabilities = self.compute_probability(self.augmentations)
        self.p = p

    # Randomly selects an Augmentation to apply to data.
    def __call__(self, image: np.ndarray, force: bool = False) -> Tuple[np.ndarray, List[Augmentation]]:
        if force or self.should_run():

            # Select one augmentation using the max value in probability values
            augmentation = self.augmentations[np.argmax(self.augmentation_probabilities)]

            # Applies the selected Augmentation.
            image = augmentation(image, force=True)
            return image, [augmentation]

    # Constructs a string containing the representations
    # of each augmentation
    def __repr__(self):
        r = "OneOf([\n"

        for augmentation in self.augmentations:
            r += f"\t{repr(augmentation)}\n"

        r += f"], p={self.p})"
        return r

    def compute_probability(self, augmentations: List[Augmentation]) -> List[float]:
        """For each Augmentation in the input list, compute the probability of applying that Augmentation.

        :param augmentations: Augmentations to compute probability list for.
        """

        # generate random 0-1 value for each augmentation
        augmentation_probabilities = [random.uniform(0, 1.0) for augmentation in augmentations]
        probability_sum = sum(augmentation_probabilities)

        # generate weighted probability by using (probability/ sum of probabilities)
        augmentation_probabilities = [
            augmentation_probability / probability_sum for augmentation_probability in augmentation_probabilities
        ]

        return augmentation_probabilities
