import cv2
import numpy as np


class SegmentationProcessor:
    def __init__(self, mask_threshold=0.1, blur_ksize=(5, 5)):
        self.mask_threshold = mask_threshold
        self.blur_ksize = blur_ksize

    def process_frame(self, frame, segmentation_mask, background_image):
        """
        Applies the background segmentation and blends the foreground onto the new background.
        """

        # 1. Smooth the segmentation mask for better blending at the edges
        mask_blur = cv2.GaussianBlur(segmentation_mask, self.blur_ksize, 0)

        # 2. Expand the mask to 3 channels (W x H x 3) to match the images
        mask_3ch = np.stack((mask_blur,) * 3, axis=-1)

        # 3. Blending Calculation (Alpha Blending):
        # Output = (Foreground * Mask) + (Background * (1 - Mask))

        # Calculate foreground (person) part:
        foreground_part = frame.astype(np.float32) * mask_3ch

        # Calculate background part:
        background_part = background_image.astype(np.float32) * (1 - mask_3ch)

        # Combine the parts
        output_image = foreground_part + background_part

        # Convert back to 8-bit integers (required for display)
        output_image = output_image.astype(np.uint8)

        return output_image