import numpy as np
from rosbags.highlevel import AnyReader

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2


import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from .idataloader import DataloaderBuilderBase, DataloaderBase

from pprint import pprint


class ROS2BagDataloaderBuilder(DataloaderBuilderBase):
    def setTopicIn(self, topic):
        self.topic = topic
        return self
    

    def setInputFile(self, input_file):
        self.input_file = input_file
        return self
    

    def usesRclpy(self):
        return False


    def build(self, perf):
        dataloader = ROS2BagDataloader(**vars(self))
        dataloader.run = perf.wrap(dataloader.run)
        return dataloader


class ROS2BagDataloader(DataloaderBase):
    @staticmethod
    def builder():
        return ROS2BagDataloaderBuilder()
    

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


    def run(self, handler):
        raise DeprecationWarning()
        with AnyReader([self.input_file]) as reader:
            connections = [x for x in reader.connections if x.topic in self.topic]
            for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
                msg = deserialize_message(rawdata, get_message(connection.msgtype))
                p = np.array(pc2.read_points(msg))
                points = np.vstack([p[k] for k in p.dtype.fields]).T
                handler.handle(points[:,:4])
                print(i)


    #def iter(self):
    def iterSequences(self, *args, **kwargs):
        def iterFrames():
            with AnyReader([self.input_file]) as reader:
                connections = [x for x in reader.connections if x.topic in self.topic]
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    msg = deserialize_message(rawdata, get_message(connection.msgtype))
                    p = np.array(pc2.read_points(msg))
                    points = np.vstack([p[k] for k in p.dtype.fields]).T
                    yield None, points[:,:4]
        yield (0, iterFrames)


    def terminate(self):
        pass