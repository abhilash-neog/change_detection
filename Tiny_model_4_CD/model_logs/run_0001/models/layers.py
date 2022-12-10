from typing import List, Optional

from torch import Tensor, reshape, stack, max, from_numpy, tensor
import torch
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
    Linear
)
import cv2
import numpy as np

class BBoxRegressor():
    def __init__(self):
        return

    def get_bboxes(self, img, max_boxes):
        #Uses opencv2 to get bounding boxes bit mask created by pixelwise classifier 
        batch_boxes = np.zeros((len(img), 4*max_boxes))
        count = 0

        #iterate through image in batch
        for i in range(len(img)):

            image = np.array(img[i].detach().cpu(), dtype = np.float)
            #round @ 0.5 to create bit mask
            image = np.round(image, 0)*255 
            image = np.array(image, dtype = np.uint8)

            if (len(image) == 1):
                image = image[0]

            boxes = np.array([])
            #find image contours in bit map
            contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if (len(contours) >1):
                contours = contours[0]

            #iterate through contours and find bounding boxes
            for cntr in contours:
                pad = 0
                x,y,w,h = cv2.boundingRect(cntr)

                if (x == 0 and y == 0 and w == 256 and h == 256): # if bounding box is entire image, discard
                    continue
                else:
                    boxes = np.append(boxes, np.array([x,y,w,h]))
                    count = count + 1
                    if (count >= max_boxes): #return if number of boxes exceeds the maximum boxes
                        break
            #add boxes to bounding boxes for the batch
            batch_boxes[i][0:len(boxes)] = boxes[0:len(boxes)]

        batch_boxes = from_numpy(batch_boxes)
        return tensor(batch_boxes, requires_grad=True)

class BoundingLinear(Module):
    #fully connected model for bounding box regression
    def __init__(self):
        super().__init__()

        #avgpool to remove stray pixels
        self._avgPool = torch.nn.AvgPool2d(3, stride=3)
        #linear layer to output 10 bounding boxes (40 = 10 * 4)
        self._linear = Linear(85*85, 40, True)

    def forward(self, x):
        x = self._avgPool(x)
        #flattening dim to go into linear layer
        x = torch.reshape(x, (x.shape[0], 85*85))
        return self._linear(x)

class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)




class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)


class MixingMaskAttentionBlock(Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class UpMask(Module):
    def __init__(
        self,
        up_dimension: int,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = Upsample(
            size=(up_dimension, up_dimension), mode="bilinear", align_corners=True
        )
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)
