import rclpy, pyrr

from .utils import point_cloud#, cylinders_markers

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Header

from builtin_interfaces.msg import Duration

from .isink import SinkBase

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation
#from buildin_interfaces.msg import Duration

import rclpy

from matplotlib.colors import to_rgba
from visualization_msgs.msg import MarkerArray, Marker
from pprint import pprint

import numpy as np


def point(*pt):
    p = Point()
    p.x = float(pt[0])
    p.y = float(pt[1])
    p.z = float(pt[2]) if len(pt) > 2 else 0.0
    return p


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
green.b = to_rgba(color)[2]
green.a = 1.0


blue = ColorRGBA()
color = "blue"
blue.r = to_rgba(color)[0]
blue.g = to_rgba(color)[1]
blue.b = to_rgba(color)[2]
blue.a = 1.0


def markersToMarkerarray(markers):
    ma = MarkerArray()
    ma.markers = markers
    return ma


def annotationToROS2Message(annotation, frame_id="map"):
    marker = Marker()
    marker.header = Header()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.time.Time().to_msg()

    marker.ns = "bounding_box"
    marker.id = annotation.track_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose = Pose()
    marker.pose.position.x = annotation.x
    marker.pose.position.y = annotation.y
    marker.pose.position.z = annotation.z
    
    marker.scale.x = annotation.length
    marker.scale.y = annotation.width
    marker.scale.z = annotation.height

    q = pyrr.Quaternion.from_z_rotation(annotation.rotation)
    marker.pose.orientation.x = q.x
    marker.pose.orientation.y = q.y
    marker.pose.orientation.z = q.z
    marker.pose.orientation.w = q.w

    marker.color = red
    marker.color.a = 0.2

    #marker.lifetime = Duration(sec=0)  # 0 means forever
    marker.frame_locked = False

    return marker



def cylinders_markers(cylinders, frame_id, stamp):
    markers = []

    c = ColorRGBA()
    color = to_rgba("red")[0]
    c.r = color
    c.g = color
    c.b = color
    c.a = 0.6
    
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


def trajectories(tracks, frame_id="map", stamp=None, namespace="", color="green"):
    markers = []
    scale = 0.1

    for id, traj in enumerate(tracks):
        marker = Marker()
        marker.ns = namespace
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.color = green if color == "green" else green
        marker.color.a = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        #marker.lifetime.nanosec = 1
        #marker.lifetime.sec = 1
        
        #marker.points = [point(pt) for pt in traj[-5:]]
        marker.points = [point(*pt) for pt in traj]
        
        markers.append(marker)
    ma = MarkerArray()
    #pprint(markers)
    ma.markers = markers
    return ma



typelookup = {
    "pointcloud": PointCloud2,
    "ellipse": MarkerArray,
    "lines": MarkerArray,
    "boundingbox": MarkerArray
}


def filter_velocity(tracks, threshold):
    raise DeprecationWarning("yeah dont use that")
    data = []
    for track in tracks:
        old = track[0]
        t = [old]
        for pos in track[1:]:
            if np.linalg.norm(np.array(pos) - np.array(old)) > threshold:
                
                t.append(pos)
            else:
                data.append(t)
                t = [pos]
            old = pos
        data.append(t)
    return data


class ROS2PublisherSink(SinkBase):
    def __init__(self):
        self.pub = {}
        self.del_marker = Marker()
        self.del_marker.action = Marker.DELETEALL

    
    def usesRclpy(self):
        return True
    
    
    def setRclpyNode(self, node):
        self.node = node
        return self


    def publish(self, name, type_, data, *args, **kwargs):
        #print(len(data), name, type_, sep="\t")
        t = rclpy.time.Time().to_msg()
        mtype = typelookup[type_]
        pub = self.pub_(name, mtype)
        if pub.get_subscription_count() <= 0:
            return

        match type_:
            case "pointcloud":
                msg = point_cloud(data, "map", *args, **kwargs)
                
            case "ellipse":
                msg = cylinders_markers(data, "map", t, *args, **kwargs)
                pub.publish(markersToMarkerarray([self.del_marker]))
                
            case "lines":
                #pprint(data)
                #data = filter_velocity(data, threshold=self.velocity_threshold)
                msg = trajectories(data, "map", t, name, *args, **kwargs)
                
            case "boundingbox":
                ma = [annotationToROS2Message(anno) for anno in data]
                msg = markersToMarkerarray(ma)
                pub.publish(markersToMarkerarray([self.del_marker]))
            case _:
                raise ValueError(f"type {type_} not found [{__file__}]")
        
        #pub = self.pub_(name, mtype)
        #if pub.get_subscription_count() > 0:
        pub.publish(msg)



    def pub_(self, name, type_):
       return self.pub.setdefault(name, self.node.create_publisher(type_, name, 10))