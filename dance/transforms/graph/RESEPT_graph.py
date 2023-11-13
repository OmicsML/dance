import cv2
import numpy as np
import scanpy as sc

from dance.data.base import Data
from dance.transforms.base import BaseTransform


class RESEPTGraph(BaseTransform):

    def __init__(self, fiducial_diameter_fullres=144.56835055243283, tissue_hires_scalef=0.150015, **kwargs):
        super().__init__(**kwargs)
        self.fiducial_diameter_fullres = fiducial_diameter_fullres
        self.tissue_hires_scalef = tissue_hires_scalef

    def scale_to_RGB(self, channel, truncated_percent):
        truncated_down = np.percentile(channel, truncated_percent)
        truncated_up = np.percentile(channel, 100 - truncated_percent)
        channel_new = ((channel - truncated_down) / (truncated_up - truncated_down)) * 255
        channel_new[channel_new < 0] = 0
        channel_new[channel_new > 255] = 255
        return np.uint8(channel_new)

    def __call__(self, data: Data) -> Data:
        xy_pixel = data.get_feature(return_type="numpy", channel="spatial_pixel", channel_type="obsm")
        sc.tl.umap(data.data, n_components=3)
        X_transform = data.get_feature(return_type="numpy", channel="X_umap", channel_type="obsm")
        X_transform[:, 0] = self.scale_to_RGB(X_transform[:, 0], 100)
        X_transform[:, 1] = self.scale_to_RGB(X_transform[:, 1], 100)
        X_transform[:, 2] = self.scale_to_RGB(X_transform[:, 2], 100)
        radius = int(0.5 * self.fiducial_diameter_fullres + 1)
        max_row = max_col = int(2000 / self.tissue_hires_scalef + 1)
        high_img = self.save_transformed_RGB_to_image_and_csv(xy_pixel[:, 0], xy_pixel[:, 1], max_row, max_col,
                                                              X_transform, plot_spot_radius=radius)
        data.data.uns[self.out] = high_img
        return data

    def save_transformed_RGB_to_image_and_csv(
        self,
        spot_row_in_fullres,
        spot_col_in_fullres,
        max_row,
        max_col,
        X_transformed,
        plot_spot_radius,
    ):

        img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255
        for index in range(len(X_transformed)):
            cv2.rectangle(
                img, (spot_col_in_fullres[index] - plot_spot_radius, spot_row_in_fullres[index] - plot_spot_radius),
                (spot_col_in_fullres[index] + plot_spot_radius, spot_row_in_fullres[index] + plot_spot_radius),
                color=(int(X_transformed[index][0]), int(X_transformed[index][1]), int(X_transformed[index][2])),
                thickness=-1)
        hi_img = cv2.resize(img, dsize=(2000, 2000), interpolation=cv2.INTER_CUBIC)
        del img, spot_row_in_fullres, spot_col_in_fullres
        return hi_img
