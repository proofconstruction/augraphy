import numpy as np

from augraphy.base.augmentation import Augmentation


class AugmentationResult:
    """Contains the result of an Augmentation's application, as well as
    the Augmentation applied. AugmentationResults are stored in an AugmentationPipeline.

    :param augmentation: The augmentation that was applied.
    :param result: The image transformed by the augmentation. Usually a numpy array.
    :param metadata: Additional data that may be added by callers.
    """

    def __init__(self, augmentation: Augmentation, result: np.ndarray, metadata=None):
        """Constructor method"""
        self.augmentation = augmentation
        self.result = result
        self.metadata = metadata
