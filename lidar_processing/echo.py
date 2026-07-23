from rclpy.node import Node
from scipy.spatial import KDTree
from .utils import PointCloud
import time
#from sensor_msgs.msg import PointCloud2
#from .utils import point_cloud, PointCloud

class Echo:
    def setParentNode(self, node:Node):
        #self.pub_echo = node.create_publisher(PointCloud2, "/lidar", 10)
        return self

    def setSink(self, sink):
        self.sink = sink
        return self

    def handle(self, points, *args, **kwargs):
        curr = PointCloud(points, KDTree(points[:,:3]))
        self.sink.publish("/lidar", "pointcloud", curr.points, axis=list("xyzi"))

        

        #if self.pub_echo.get_subscription_count() > 0:
        #    pcd = point_cloud(curr.points[:3,:], "map", "xyzidbc")
        #    self.pub_echo.publish(pcd)