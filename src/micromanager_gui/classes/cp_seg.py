import tensorflow as tf
import tensorflow.keras.backend as K
from pathlib import Path
import numpy as np
import random
import os
from skimage import feature, filters, morphology, segmentation
from scipy import ndimage as ndi

class SegmentNeurons():
    """Segment neurons in the image."""
    def __init__(self) -> None:
        self._model: tf.keras.models = None

    def _load_model(self, img_size: int) -> None:
        """Load the model"""
        # only initiate the trained model once
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, f'unet_calcium_{img_size}.hdf5')
        self._model = tf.keras.models.load_model(path, custom_objects={"K": K})
        
    def _segment(self, img_stack, minsize, background_label):
        '''
        Predict the cell bodies using the trained NN model
        and further segment after removing small holes and objects

        Parameters:
        -------------
        img_stack: ndarray. shape=(# of frames, # of rows, # of colomns) array of the images
        minsize: float or int. the threshold to determine if the predicted cell body is too small
        background_label: int. the value set to be the background

        Return:
        -------------
        labels: ndarray. shape= shape of img_stack. a labeled matrix of segmentations of the same type as markers
        label_layer: ndarray. the label/segmentation layer image
        roi_dict: dict. a dictionary of label-position pairs
        '''
        img_norm = np.max(img_stack, axis=0) / np.max(img_stack)
        img_predict = self._model.predict(img_norm[np.newaxis, :, :])[0, :, :]

        if np.max(img_predict) > 0.3:
            # use Otsu's method to find the cooridnates of the cell bodies
            th = filters.threshold_otsu(img_predict)
            img_predict_th = img_predict > th
            img_predict_remove_holes_th = morphology.remove_small_holes(img_predict_th, area_threshold=minsize * 0.3)
            img_predict_filtered_th = morphology.remove_small_objects(img_predict_remove_holes_th, min_size=minsize)
            distance = ndi.distance_transform_edt(img_predict_filtered_th)
            local_max = feature.peak_local_max(distance,
                                               min_distance=10,
                                               footprint=np.ones((15, 15)),
                                               labels=img_predict_filtered_th)

            # create masks over the predicted cell bodies and add a segmentation layer
            local_max_mask = np.zeros_like(img_predict_filtered_th, dtype=bool)
            local_max_mask[tuple(local_max.T)] = True
            markers = morphology.label(local_max_mask)
            labels = segmentation.watershed(-distance, markers, mask=img_predict_filtered_th)
            roi_dict, labels = self._getROIpos(labels, background_label)

        return labels, roi_dict

    def _check_model(self):
        """Check if the model is loaded."""
        return self._model is not None

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

        area_dict, roi_to_delete = self.get_ROI_area(roi_dict, 100)

        # delete roi in label layer and dict
        for r in roi_to_delete:
            coords_to_delete = np.array(roi_dict[r]).T.tolist()
            masks[tuple(coords_to_delete)] = 0
            roi_dict[r] = []

        # move roi in roi_dict after removing some labels
        for r in range(1, (len(roi_dict) - len(roi_to_delete) + 1)):
            i = 1
            while not roi_dict[r]:
                roi_dict[r] = roi_dict[r + i]
                roi_dict[r + i] = []
                i += 1

        # delete extra roi keys
        for r in range((len(roi_dict) - len(roi_to_delete) + 1), (len(roi_dict) + 1)):
            del roi_dict[r]

        # update label layer with new roi
        for r in roi_dict:
            roi_coords = np.array(roi_dict[r]).T.tolist()
            masks[tuple(roi_coords)] = r

        return roi_dict, masks

    def get_ROI_area(self, roi_dict: dict, threshold: float):
        '''
        to get the areas of each ROI in the ROI_dict

        Parameters:
        ------------
        roi_dict: dict. the dictionary of the segmented ROIs
        threshold: float or int. The value below which the segmentation would be considered small

        Returns:
        -----------
        area: dict. a dictionary of the length of the coordinates (?) in the roi_dict
        small_roi: list. a list of small rois if their coordinates are smaller than the set threshold
        '''
        area = {}
        small_roi = []
        for r in roi_dict:
            area[r] = len(roi_dict[r])
            if area[r] < threshold:
                small_roi.append(r)
        return area, small_roi
    
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
        labels, roi_dict = self._segment(img[10], file_path)
        raw_signal = self._ROI_intensity(roi_dict, img)
        dff, median, bg = self._calculate_DFF(raw_signal)

        return roi_dict, raw_signal, dff
