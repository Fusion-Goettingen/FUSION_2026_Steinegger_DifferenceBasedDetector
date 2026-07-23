import numpy as np
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import sensor_msgs_py.point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header
#from buildin_interfaces.msg import Duration

import rclpy

from matplotlib.colors import to_rgba
from visualization_msgs.msg import MarkerArray, Marker
from pprint import pprint
from collections import namedtuple
from builtin_interfaces.msg import Duration

from std_msgs.msg import ColorRGBA

from matplotlib.colors import to_rgba

Ellipse = namedtuple("Ellipse", ["xyz", "eigval", "psi", "dim"])




red = ColorRGBA()
    
color = "red"
red.r = to_rgba(color)[0]
red.g = to_rgba(color)[1]
red.b = to_rgba(color)[2]
red.a = 1.0



green = ColorRGBA()
    
color = "green"
green.r = to_rgba(color)[0]
green.g = to_rgba(color)[1]
green.b = to_rgba(color)[2] + 50
green.a = 0.1




def clustersToEllipses(clusters, dim):
    repr = []
    for cluster in clusters:
        if cluster.shape[1] < 4:
            print(f"[Warning] Cluster with only {cluster.shape[1]} elements ignored.")
            continue
        #print(cluster.shape)
        clu = cluster[:dim,:]
        m = clu.mean(axis=1)
        #mm = cluster[:,:dim] - m[:,np.newaxis]
        C = np.cov(clu)
        # print(cluster.shape)
        #C = (mm @ mm.T) / (cluster.shape[1]-1)
        eig_vals, eig_vecs = np.linalg.eig(C)
        psi = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
        #print(eig_vals)
        eigval = [3*np.sqrt(ev) for ev in eig_vals] # *2 -> 95%
        c = Ellipse(m, eigval, psi, dim)
        
        repr.append(c)
    return repr



def covarmat_to_ellipse(ext, pos, dim):
    repr = []
    for C, m in zip(ext, pos):
        eig_vals, eig_vecs = np.linalg.eigh(C)
        psi = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
        
        eigval = [3*np.sqrt(ev) for ev in eig_vals] # *2 -> 95%
        e = Ellipse(m, eigval, psi, dim)
        
        repr.append(e)
    return repr



def point_cloud(points, parent_frame, axis):
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.
    
    fields = [sensor_msgs.PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, (_, n) in enumerate(zip(range(points.shape[1]), axis))]
    
    header = std_msgs.Header()
    header.frame_id = parent_frame

    #pcd = pc2.create_cloud(header, fields, points.tolist())
    pcd = pc2.create_cloud(header, fields, points)
    return pcd



PointCloud = namedtuple("PointCloud", ["points", "tree"])

def point(pt):
    p = Point()
    p.x = float(pt[0])
    p.y = float(pt[1])
    p.z = float(pt[2])
    return p



def marker(points_list, frame_id="map", ns="points", marker_id=0, color=(1, 1, 1)):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.time.Time().to_msg()
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    marker.scale.x = 0.1
    marker.scale.y = 0.1

    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = 1.0

    # Convert input points to geometry_msgs/Point and add to marker
    marker.points = [point(pt) for pt in points_list]
    
    return marker






def pathlines(tracks,  frame_id="map", stamp=None, oldids=[]):
    markers = []

    c = ColorRGBA()
    
    color = "red"
    c.r = to_rgba(color)[0]
    c.g = to_rgba(color)[1]
    c.b = to_rgba(color)[2]
    c.a = 1.0 #to_rgba(color)[3] #if alpha is None else float(alpha)
    #marker_msg.color = c

    for id, track in enumerate(tracks):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = id
        marker.type = Marker.CUBE_LIST#Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.color = c
        marker.scale.x = .4
        marker.scale.y = .4
        marker.scale.z = .4
        #marker.lifetime.nanosec = 1
        marker.lifetime.sec = 10

        p = Point()
        p.x = track[0]
        p.y = track[1]
        p.z = track[2] if len(track) == 3 else 1.0
        marker.points.append(p)
        
        #for point in track:
        #    p = Point()
        #    p.x = float(point[0])
        #    p.y = float(point[1])
        #    p.z = float(point[2])
        #    marker.points.append(p)

        
        markers.append(marker)
    ma = MarkerArray()

    ma.markers = markers
    return ma



def ellipse_markers(ellipses, frame_id, stamp):
    markers = []

    c = ColorRGBA()
    color = to_rgba("red")[0]
    c.r = color
    c.g = color
    c.b = color
    c.a = 0.2
    
    for id, ellipse in enumerate(ellipses):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.color = c
        marker.header.stamp = stamp
        marker.id = id
        marker.type = Marker.SPHERE if ellipse.dim == 3 else Marker.CYLINDER
        marker.action = 0#Marker.MODIFY
        marker.lifetime.sec = 10

        marker.pose.position.x = float(ellipse.xyz[0])
        marker.pose.position.y = float(ellipse.xyz[1])
        marker.pose.position.z = float(ellipse.xyz[2]) if ellipse.dim == 3 else 0.0

        marker.scale.x = ellipse.eigval[0]
        marker.scale.y = ellipse.eigval[1]
        marker.scale.z = ellipse.eigval[2] if ellipse.dim == 3 else 1.0

        orientation = Rotation.from_euler("XYZ", [0, 0, ellipse.psi], degrees=False).as_quat()
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]
        
        markers.append(marker)

    ma = MarkerArray()
    ma.markers = markers
    return ma



def cylinders_markers(cylinders, frame_id, stamp):
    markers = []

    c = ColorRGBA()
    color = to_rgba("red")[0]
    c.r = color
    c.g = color
    c.b = color
    c.a = 0.8
    
    for id, cylinder in enumerate(cylinders):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.color = c
        marker.header.stamp = stamp
        marker.id = id
        marker.type = Marker.CYLINDER
        marker.action = 0#Marker.MODIFY
        marker.lifetime.sec = 10

        marker.pose.position.x = float(cylinder.xyz[0])
        marker.pose.position.y = float(cylinder.xyz[1])
        marker.pose.position.z = float(cylinder.xyz[2]) if cylinder.dim == 3 else 0.0

        marker.scale.x = cylinder.eigval[0]
        marker.scale.y = cylinder.eigval[1]
        marker.scale.z = cylinder.eigval[2] if cylinder.dim == 3 else 1.0

        orientation = Rotation.from_euler("XYZ", [0, 0, cylinder.psi], degrees=False).as_quat()
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]
        
        markers.append(marker)

    ma = MarkerArray()
    ma.markers = markers
    return ma



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




