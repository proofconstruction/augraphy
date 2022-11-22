import random


class Augmentation:
    """The base class which all pipeline augmentations inherit from.

    :param p: The probability that this augmentation will be run when executed as part of a pipeline.
    """

    def __init__(self, p: float = 0.5):
        """Constructor method"""
        self.p = p

    def should_run(self) -> bool:
        """Determines whether or not the augmentation should be applied
        by callers.
        """
        return random.uniform(0.0, 1.0) <= self.p
