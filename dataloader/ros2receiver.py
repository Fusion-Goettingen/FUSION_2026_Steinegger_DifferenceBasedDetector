import numpy as np, rclpy
from sensor_msgs.msg import PointCloud2
from rclpy.node import Node

import sensor_msgs_py.point_cloud2 as pc2

from .idataloader import DataloaderBuilderBase, DataloaderBase

class ROS2DataloaderBuilder(DataloaderBuilderBase):
    def setTopicIn(self, topic):
        self.topic = topic
        return self


    def setRclpyNode(self, node):
        self.node = node
        return self


    def build(self, perf):
        return ROS2Dataloader(perf, topic=self.topic, node = self.node)
    
    
    def usesRclpy(self):
        return True
    

    def setRclpyNode(self, node):
        self.node = node
        return self
    

class ROS2Dataloader(DataloaderBase):
    def __init__(self, perf, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        callback = perf.wrap(self.callback)
        self.sub = self.node.create_subscription(PointCloud2, self.topic, callback, 10)


    @staticmethod
    def builder():
        return ROS2DataloaderBuilder()


    def callback(self, msg):
        p = pc2.read_points(msg)
        points = np.vstack([p[k] for k in p.dtype.fields]).T

        self.handler.handle(points)


    def run(self, handler):
        self.handler = handler

        try:
            handler.setParentNode(self)
        except AttributeError:
            pass
    
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.terminate()


    def terminate(self):
        self.node.destroy_node()