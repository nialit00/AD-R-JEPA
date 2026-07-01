import os
from functools import reduce

import numpy as np
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix




def get_radar_data(nusc, sample_rec, nsweeps, min_distance, use_radar_filters, dataroot):
    """
    Returns at most nsweeps of radar in the ego frame.
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # import ipdb; ipdb.set_trace()

    # points = np.zeros((5, 0))
    points = np.zeros((19, 0))  # 18 plus one for time

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['RADAR_FRONT']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    if use_radar_filters:
        RadarPointCloud.default_filters()
    else:
        RadarPointCloud.disable_filters()

    # Aggregate current and previous sweeps.
    # from all radars
    radar_chan_list = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    for radar_name in radar_chan_list:
        sample_data_token = sample_rec['data'][radar_name]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = RadarPointCloud.from_file(os.path.join(dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            # print('time_lag', time_lag)
            # print('new_points', new_points.shape)

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points