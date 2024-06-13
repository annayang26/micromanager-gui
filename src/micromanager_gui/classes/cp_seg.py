from cellpose import io, models, plot
from pathlib import Path
import numpy as np
import random
import os

class SegmentNeurons():
    """Segment neurons in the image."""
    def __init__(self) -> None:
        self._model: models.CellposeModel = None

    def _load_model(self) -> None:
        """Load the model"""
        print("_________Loading cellpose model__________")
        dir_path = Path(__file__).parent
        model_path = Path.joinpath(dir_path, "CP_calcium")
        self._model = models.CellposeModel(gpu=False,
                                                pretrained_model=model_path)
        
    def _segment(self, img: np.ndarray, path: str, channels: list = [0,0]) -> np.ndarray:
        """Segmetnt the img."""
        masks, flows, _ = self._model.eval(img,
                                        diameter=None,
                                        flow_threshold=0.1,
                                        cellprob_threshold=0,
                                        channels=channels)

        if not os.path.isdir(path):
            os.mkdir(path)

        self._save_overlay(img, channels, masks, path)
        rgb_mask = plot.mask_rgb(masks)
        save_path = os.path.join(path, "masks.tif")
        io.save_masks(img, rgb_mask, flows, save_path, tif=True)

        return masks

    def _check_model(self):
        """Check if the model is loaded."""
        return self._model is not None

    def _save_overlay(self, img: np.ndarray, channels: list, 
                      masks: np.ndarray, save_path: str) -> None:
        """Save the overlay image of masks over original image."""
        img0 = img.copy()

        if img0.shape[0] < 4:
            img0 = np.transpose(img0, (1, 2, 0))
        if img0.shape[-1] < 3 or img0.ndim < 3:
            img0 = plot.image_to_rgb(img0, channels=channels)
        else:
            if img0.max() <= 50.0:
                img0 = np.uint8(np.clip(img0, 0, 1) * 255)

        # generate mask over original image
        overlay = plot.mask_overlay(img0, masks)
        io.imsave(os.path.join(save_path, "overlay.jpg"), overlay)

    def _getROIpos(self, masks: list | np.ndarray, bg_label: int = 0) -> tuple[dict, np.ndarray]:
        """Get pos of ROIs"""
        # sort the labels and filter the unique ones
        u_labels = np.unique(masks)

        # create a dict for the labels
        roi_dict = {}
        for u in u_labels:
            roi_dict[u.item()] = []

        # record the coordinates for each label
        for x in range(masks.shape[0]):
            for y in range(masks.shape[1]):
                roi_dict[masks[x, y]].append([x, y])

        # delete any background labels
        del roi_dict[bg_label]

        # area_dict, roi_to_delete = self.get_ROI_area(roi_dict, 100)

        # # delete roi in label layer and dict
        # for r in roi_to_delete:
        #     coords_to_delete = np.array(roi_dict[r]).T.tolist()
        #     masks[tuple(coords_to_delete)] = 0
        #     roi_dict[r] = []

        # # move roi in roi_dict after removing some labels
        # for r in range(1, (len(roi_dict) - len(roi_to_delete) + 1)):
        #     i = 1
        #     while not roi_dict[r]:
        #         roi_dict[r] = roi_dict[r + i]
        #         roi_dict[r + i] = []
        #         i += 1

        # # delete extra roi keys
        # for r in range((len(roi_dict) - len(roi_to_delete) + 1), (len(roi_dict) + 1)):
        #     del roi_dict[r]

        # # update label layer with new roi
        # for r in roi_dict:
        #     roi_coords = np.array(roi_dict[r]).T.tolist()
        #     masks[tuple(roi_coords)] = r

        return roi_dict, masks
    
    def _ROI_intensity(self, roi_dict: dict, img_stack: np.ndarray) -> dict:
        """Calculate raw signal averaged across all pixels in the ROI."""
        f = {}
        for r in roi_dict:
            f[r] = np.zeros(img_stack.shape[0])
            roi_coords = np.array(roi_dict[r]).T.tolist()
            for z in range(img_stack.shape[0]):
                img_frame = img_stack[z, :, :]
                f[r][z] = np.mean(img_frame[tuple(roi_coords)])
        return f
    
    def _calculate_DFF(self, roi_signal: dict) -> tuple[dict, dict, dict]:
        """Calculate ∆F/F."""
        dff = {}
        median = {}
        bg = {}
        for n in roi_signal:
            background, median[n] = self._calculate_bg(roi_signal[n], 200)
            bg[n] = background.tolist()
            dff[n] = (roi_signal[n] - background) / background
            dff[n] = dff[n] - np.min(dff[n])
        return dff, median, bg
    
    def _calculate_bg(self, raw_f: dict, window: int) -> tuple[np.ndarray, list]:
        """Calculate background intensity."""
        background = np.zeros_like(raw_f)
        background[0] = raw_f[0]
        median = [background[0]]
        for y in range(1, len(raw_f)):
            x = y - window
            if x < 0:
                x = 0
            lower_quantile = raw_f[x:y] <= np.median(raw_f[x:y])
            background[y] = np.mean(raw_f[x:y][lower_quantile])
            median.append(np.median(raw_f[x:y]))
        return background, median

    def _run(self, img: np.ndarray, file_path: str) -> tuple[dict, dict]:
        """Run the entire segmentation process."""
        print("__________Segmenting___________")
        masks = self._segment(img[100, :, :], file_path)
        roi_dict, masks = self._getROIpos(masks, 0)
        raw_signal = self._ROI_intensity(roi_dict, img)
        dff, median, bg = self._calculate_DFF(raw_signal)

        return roi_dict, raw_signal, dff

    def _run_evk(self):
        """Run segmentation for evoked activity."""
        #TODO: to be implemented