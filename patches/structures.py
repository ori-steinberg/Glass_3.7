from __future__ import division
from detectron2.structures import ImageList
import mock

def ImageList__len(self):
    return self.image_sizes.shape[0]

patches = (
    mock.patch.object(ImageList, '__len__', ImageList__len),
)