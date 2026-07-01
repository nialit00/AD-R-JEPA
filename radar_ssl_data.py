import torch
import numpy as np

from utils.radar_utils import get_radar_data


class RadarSSLData(torch.utils.data.Dataset):
    """
    Clean Radar-only SSL dataset for JEPA pretraining.
    Uses BEVCar radar + voxelization pipeline but avoids full camera pipeline.
    """

    def __init__(
        self,
        nusc,
        vox_util,
        nsweeps=5,
        Z=1,
        Y=200,
        X=200,
        use_shallow_metadata=False,
        use_radar_occupancy_map=False
    ):
        self.nusc = nusc
        self.vox_util = vox_util

        self.nsweeps = nsweeps
        self.use_shallow_metadata = use_shallow_metadata
        self.use_radar_occupancy_map = use_radar_occupancy_map

        self.Z = Z
        self.Y = Y
        self.X = X

        self.samples = list(nusc.sample)
        self.indices = list(range(len(self.samples)))

    def __len__(self):
        return len(self.indices)

    def get_radar_data(self, rec):
        """
        Reuse BEVCar logic indirectly via nuscenes format.
        Minimal version: expects BEVCar-style radar loader to exist in project.
        """

        radar_data = get_radar_data(
            nusc=self.nusc,
            sample_rec=rec,
            nsweeps=self.nsweeps,
            min_distance=2.2,
            use_radar_filters=False,
            dataroot=self.nusc.dataroot
        )

        return radar_data

    def __getitem__(self, index):

        rec = self.samples[index]

        # -----------------------------
        # RADAR ONLY
        # -----------------------------
        radar_data = self.get_radar_data(rec)

        radar_data = np.transpose(radar_data)

        V = 700 * self.nsweeps

        if radar_data.shape[0] < V:
            radar_data = np.pad(
                radar_data,
                [(0, V - radar_data.shape[0]), (0, 0)],
                mode='constant'
            )

        radar_data = torch.from_numpy(radar_data).float()

        # -----------------------------
        # VOXELIZATION (BEVCar compatible)
        # -----------------------------

        rad_data = radar_data.permute(0, 1)  # R, C

        xyz_rad = rad_data[:, :3].unsqueeze(0)

        # WICHTIG: erst ALLES behalten, dann sauber auf 4 Radar-Features reduzieren
        raw_meta = rad_data[:, 3:]

        # feste Auswahl (nicht mehr experimentieren!)
        meta_rad = raw_meta[:, :4].unsqueeze(0)

        velo_T_velo = torch.eye(4).unsqueeze(0)

        rad_xyz_cam0 = xyz_rad  # already in ego frame

        #print(rad_data.shape)
        voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels = \
            self.vox_util.voxelize_xyz_and_feats_voxelnet(
                rad_xyz_cam0,
                meta_rad,
                self.Z,
                self.Y,
                self.X,
                assert_cube=False,
                use_radar_occupancy_map=self.use_radar_occupancy_map
            )

        voxel_input_feature_buffer = voxel_input_feature_buffer.squeeze(0)
        voxel_coordinate_buffer = voxel_coordinate_buffer.squeeze(0)
        number_of_occupied_voxels = number_of_occupied_voxels.squeeze(0)

        return {
            "voxel_input_feature_buffer": voxel_input_feature_buffer,
            "voxel_coordinate_buffer": voxel_coordinate_buffer,
            "number_of_occupied_voxels": number_of_occupied_voxels,
        }