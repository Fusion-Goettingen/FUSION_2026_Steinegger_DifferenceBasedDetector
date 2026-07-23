## From Simon -> Sick N8dW

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from matplotlib.colors import to_rgba


def time_to_float(t):
    return t.sec + t.nanosec * 1e-9


def pointcloud2_to_numpy(pc2_msg) -> np.ndarray:
    # read points from pc2_msg, returning a generator object
    point_generator = pc2.read_points(pc2_msg)

    # contains x - y - z - intensity - index
    cloud = np.array([list(p) for p in point_generator])

    return cloud[:, :4]  # drop index for return


def aligned_bbox_to_cube_msg(box, header, marker_id, alpha):
    marker_msg = Marker()
    marker_msg.header = header
    marker_msg.id = marker_id
    marker_msg.type = Marker.CUBE
    marker_msg.action = Marker.MODIFY

    marker_msg.pose.position.x = box[0]
    marker_msg.pose.position.y = box[1]
    marker_msg.pose.position.z = box[2]

    marker_msg.scale.x = box[3]
    marker_msg.scale.y = box[4]
    marker_msg.scale.z = box[5]
    c = ColorRGBA()
    color = f"C{marker_id}"
    color = "red"
    c.r = to_rgba(color)[0]
    c.g = to_rgba(color)[1]
    c.b = to_rgba(color)[2]
    c.a = to_rgba(color)[3] if alpha is None else float(alpha)
    marker_msg.color = c
    return marker_msg


def cylinder_to_marker_msg(cylinder, header, marker_id, alpha):
    marker_msg = Marker()
    marker_msg.header = header
    marker_msg.id = marker_id
    marker_msg.type = Marker.CYLINDER
    marker_msg.action = Marker.MODIFY

    marker_msg.pose.position.x = cylinder[0]
    marker_msg.pose.position.y = cylinder[1]
    marker_msg.pose.position.z = cylinder[2]

    marker_msg.scale.x = cylinder[3]
    marker_msg.scale.y = cylinder[4]
    marker_msg.scale.z = cylinder[5]

    orientation = Rotation.from_euler("XYZ", [0, 0, cylinder[6]], degrees=False).as_quat()
    marker_msg.pose.orientation.x = orientation[0]
    marker_msg.pose.orientation.y = orientation[1]
    marker_msg.pose.orientation.z = orientation[2]
    marker_msg.pose.orientation.w = orientation[3]

    c = ColorRGBA()
    color = f"C{marker_id}"
    color = "red"
    c.r = to_rgba(color)[0]
    c.g = to_rgba(color)[1]
    c.b = to_rgba(color)[2]
    c.a = to_rgba(color)[3] if alpha is None else float(alpha)
    marker_msg.color = c

    return marker_msg


def aligned_bbox_to_markerarray(data,
                                header,
                                alpha=0.1,
                                marker_type="bbox") -> MarkerArray:
    marker_list = []
    for i, box in enumerate(data):
        if marker_type == "bbox":
            marker_list.append(aligned_bbox_to_cube_msg(box, header, i, alpha))  # TODO LineStrip would look better
        elif marker_type == "cylinder":
            marker_list.append(cylinder_to_marker_msg(box, header, i, alpha))
        else:
            raise NotImplementedError(f"Marker type '{marker_type}' not implemented!")
    ma_msg = MarkerArray()
    ma_msg.markers = marker_list
    return ma_msg
