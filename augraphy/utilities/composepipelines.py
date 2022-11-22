from typing import List

import numpy as np

from augraphy.base.augmentationpipeline import AugraphyPipeline


class ComposePipelines:
    """The composition of multiple AugraphyPipelines.
    Define AugraphyPipelines elsewhere, then use this to compose them.
    ComposePipelines objects are callable on images (as numpy.ndarrays).

    :param pipelines: A list contains multiple augraphy.base.AugraphyPipeline.
    """

    def __init__(self, pipelines: List[AugraphyPipeline]):
        self.pipelines = pipelines

    def __call__(self, image: np.ndarray) -> dict:

        augmented_image = image.copy()
        newpipeline = dict()

        for i, pipeline in enumerate(self.pipelines):
            data_output = pipeline.augment(augmented_image)
            augmented_image = data_output["output"]

            for key in data_output.keys():
                newkey = "pipeline" + str(i) + "-" + key
                newpipeline[newkey] = data_output[key]

        return newpipeline
