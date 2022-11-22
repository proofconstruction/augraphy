from typing import Callable
from typing import List
from typing import Union

import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class Function(Augmentation):
    """Accepts an arbitrary function or list of functions to apply in the pipeline.

    :param fs: The function or list of functions to apply.
    """

    def __init__(self, fs: Union[List[Callable], Callable], p: float = 1):
        self.fs = fs
        super().__init__(p=p)

    def applyFs(self, fs: Union[List[Callable], Callable], img: np.ndarray) -> np.ndarray:
        """Applies any fs to img sequentially."""
        current = img

        if type(fs) == list:
            for f in fs:
                current = f(current)

        else:
            current = fs(current)

        return current

    def __call__(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        image = image.copy()
        output = self.applyFs(self.fs, image)

        return output
