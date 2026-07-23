import numpy as np, sys, pypatchworkpp, scipy, numba#, icp_cpp
from sklearn.cluster import DBSCAN
from .utils import PointCloud, filterByFrameByFrameVelocity


from scipy.spatial import KDTree
from collections import namedtuple


sys.path.append('../random_matrix_tracker')
from random_matrix_tracker import RandomMatrixTracker
sys.path.append('../global_nearest_neighbor')
from global_nearest_neighbor import GlobalNearestNeighborTrackManager

Ellipse = namedtuple("Ellipse", ["xyz", "eigval", "psi", "dim"])

def extendToEllipse(pos, ext):
    ell = []
    for m, C in zip(pos, ext):
        eig_vals, eig_vecs = np.linalg.eig(C)
        psi = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
        eigval = [3*np.sqrt(ev) for ev in eig_vals]
        c = Ellipse(m, eigval, psi, 2)
        ell.append(c)
    return ell


@numba.njit(cache=False)
def dist(one, two, indices):
    d = np.empty((len(two)))
    for i, (a_, b_) in enumerate(zip(one[indices.ravel()], two)):
        a_ = a_[:3]
        b_ = b_[:3]
        d[i] = np.linalg.norm(b_) - np.linalg.norm(a_)
    return  d[:,None]


class Baseline:
    def __init__(self):
        self.histsize = None
        self.dim = 2

        self.last = None
        self.curr = None
        
        self.prev = None
        self.trail = []
    

    def fromParameters(self, **kwargs):
        self.setGate(kwargs["gate"])
        self.setDBSCAN(kwargs["eps"], kwargs["min_samples"])
        self.setHistorySize(kwargs["histsize"])
        

    def finish(self):
        if self.dim == 3:
            raise ValueError("Not implemented")
            self.initTracker3D()
        
        elif self.dim == 2:
            self.initTracker2D()
        
        else:
            raise ValueError("dimension needs to be 2")

        params = pypatchworkpp.Parameters()
        params.sensor_height = 4.0

        self.grm = pypatchworkpp.patchworkpp(params)

        return self
    

    def initTracker2D(self):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]]).astype(float)
        
        P = np.eye(4).astype(float)
        Q = np.diag([1, 1, 1, 1])
        R = np.eye(2).astype(float) * 0.0005 # precision
        F = lambda T: np.array([[1, 0, T, 0],
                                [0, 1, 0, T],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).astype(float)

        self.tracker = RandomMatrixTracker(H, P, R, F, Q, tau=10, z=0.25)
        self.tracker_manager = GlobalNearestNeighborTrackManager(self.tracker, H, history_size=self.histsize, gate=self.gate)

        x = np.array([[0, 0, 1, 1]]).T.astype(float)
        X = np.eye(2) * 0.04
        P = np.eye(4).astype(float)
        alpha = 20

        self.tracker.setPrior(x, X, P, alpha)

        return self


    def setQueryRadius(self, radius):
        self.qradius = radius
        return self


    def setGate(self, gate):
        self.gate = gate
        return self
    

    def setDBSCAN(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        return self
    

    def setHistorySize(self, size):
        self.histsize = size
        return self


    def setSink(self, sink):
        self.sink = sink
        return self


    def remove_ground(self, points):
        self.grm.estimateGround(points)
        points = self.grm.getNonground()
        return points
    

    def smooth(self, curr, points):
        if len(points) == 0:
            return np.zeros((0, 4))
        smooth = curr.tree.query_ball_point(points[:,:3], self.qradius)
        smoothB = np.unique(np.hstack(smooth))
        return curr.points[smoothB]

    
    def clustering(self, points, eps, min_samples):
        if len(points) == 0:
            return np.zeros((0, 4)), []
        
        dbscan = DBSCAN(eps, min_samples=min_samples).fit(points[:,:3])
        labels = dbscan.labels_.reshape((-1, 1))
        pts = points[labels[:,0] != -1,:3]
        labels = labels[labels[:,0] != -1]

        points = np.hstack([pts, labels])
        clu = [pts[labels[:,0] == lbl].T for lbl in np.unique(labels)]
        return points, clu


    def tracking(self, clusters):
        dim = self.dim
        clus = [c[:dim,:] for c in clusters if c.shape[1] > 4] # more than 4 points
        T = 1
        self.tracker_manager.match(clus, T=T)
        tracks = [track for track in self.tracker_manager.tracks if track.isAlive()]
        
        traj, ext = filterByFrameByFrameVelocity(tracks, self.velocity_threshold)
        return traj, None, ext, tracks


    def handle(self, points):
        self.sink.publish("/lidar/all", "pointcloud", points[:,:4], axis="xyzi")
        points = points[:,:4]

        points = self.remove_ground(points)
        self.sink.publish("/lidar/nonGround", "pointcloud", points[:,:3], axis="xyzi")

        curr = PointCloud(points, KDTree(points[:,:3]))
        if self.last is None:
            self.last = curr
            return
        
        points, clusters = self.clustering(points, self.eps, self.min_samples)
        
        self.sink.publish("/dbscan", "pointcloud", points, axis=list("xyzi") + ["labels"])

        pos, pre, ext, tracks = self.tracking(clusters)
        
        
        self.sink.publish("/tracker/position", "lines", pos, color="red")
        self.sink.publish("/tracker/extend", "ellipse", extendToEllipse([e[0] for e in ext], [e[1] for e in ext]))
        self.last = curr