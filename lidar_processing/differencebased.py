import numpy as np
import sys, pypatchworkpp, numba
from sklearn.cluster import DBSCAN
from .utils import PointCloud, filterByFrameByFrameVelocity

from .utils import extendToEllipse

from scipy.spatial import KDTree
from sklearn.neighbors import BallTree


sys.path.append('../random_matrix_tracker')
from random_matrix_tracker import RandomMatrixTracker
sys.path.append('../global_nearest_neighbor')
from global_nearest_neighbor import GlobalNearestNeighborTrackManager


@numba.njit(cache=True)
def thetaV(vv, w):
    return np.array([np.arctan2(v[0]*w[1] - v[1]*w[0], v[0]*w[0] + v[1]*w[1]) for v in vv])


@numba.njit(cache=True)
def dist(one, two, indices):
    print("dist")
    d = np.empty((len(two)))
    for i, (a_, b_) in enumerate(zip(one[indices.ravel()], two)):
        a_ = a_[:3]
        b_ = b_[:3]
        d[i] = np.linalg.norm(b_) - np.linalg.norm(a_)
    return  d[:,None]


def cartesianToSpherical(points):
    depth = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(points[:,1], points[:,0])
    elevation = np.arccos(points[:,2]/ depth)
    return np.vstack([azimuth, elevation, depth]).T


class DiffTracker:
    def __init__(self):
        self.histsize = None
        self.dim = 2

        self.last = None
        self.curr = None
        
        self.prev = None
        self.trail = []
    

    def fromParameters(self, **kwargs):
        self.setThreshold(kwargs["threshold"])
        self.setQueryRadius(kwargs["query_radius"])
        self.setGate(kwargs["gate"])
        self.setDBSCAN1(kwargs["eps1"], kwargs["min_samples1"])
        self.setDBSCAN2(kwargs["eps2"], kwargs["min_samples2"])
        self.setHistorySize(kwargs["histsize"])
        self.setRemoveGround(kwargs["rmground"])
        self.setBoundaryCheckRadius(kwargs["boundary_check_radius"])
        
        
    def finish(self):
        if self.dim == 3:
            raise ValueError("dim = 3, not supported")
            self.initTracker3D()
        
        elif self.dim == 2:
            self.initTracker2D()
        
        else:
            raise ValueError("dimension needs to be 2")

        params = pypatchworkpp.Parameters()
        #params.verbose = True
        params.sensor_height = 4.0

        self.grm = pypatchworkpp.patchworkpp(params)

        return self
    

    def initTracker2D(self):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]]).astype(float)
        
        P = np.eye(4).astype(float)
        #Q = np.diag([0.5, 0.5, 0.5, 0.5])
        Q = np.diag([1, 1, 1, 1])
        R = np.eye(2).astype(float) * 0.0005 # precision
        F = lambda T: np.array([[1, 0, T, 0],
                                [0, 1, 0, T],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).astype(float)

        self.tracker = RandomMatrixTracker(H, P, R, F, Q, tau=10, z=0.25)
        self.tracker_manager = GlobalNearestNeighborTrackManager(self.tracker, H, history_size=self.histsize, gate=self.gate)

        x = np.array([[0, 0, 1, 1]]).T.astype(float)
        #gt = np.diag([4, 1])
        X = np.eye(2) * 0.04
        P = np.eye(4).astype(float)
        alpha = 20

        self.tracker.setPrior(x, X, P, alpha)

        return self


    def setThreshold(self, threshold):
        self.threshold = threshold
        return self


    def setQueryRadius(self, radius):
        self.qradius = radius
        return self
        

    def setGate(self, gate):
        self.gate = gate
        return self
    

    def setDBSCAN1(self, eps, min_samples):
        self.eps1 = eps
        self.min_samples1 = min_samples
        return self
    
    
    def setDBSCAN2(self, eps, min_samples):
        self.eps2 = eps
        self.min_samples2 = min_samples
        return self
    

    def setHistorySize(self, size):
        self.histsize = size
        return self


    def setSink(self, sink):
        self.sink = sink
        return self
    

    def setRemoveGround(self, rmg):
        self.rmground = rmg
        return self

    def setBoundaryCheckRadius(self, radius):
        self.boundarycheck_radius = radius
        return self
    

    def remove_ground(self, points):
        self.grm.estimateGround(points)
        points = self.grm.getNonground()
        return points
    

    def difference(self, curr, last, threshold):
        f, _ = last.tree.query(curr.points[:,:3], 1)
        b, _ = curr.tree.query(last.points[:,:3], 1)
        return last.points[b>threshold], curr.points[f>threshold], np.hstack([last.points, b[:,None]]), np.hstack([curr.points, f[:,None]])


    def angle_difference(self, A, B):
        if len(A) == 0 or len(B) == 0:
            return np.zeros((0, 5))
        
        one = cartesianToSpherical(A[:,:3])
        tree = BallTree(one[:,:2])

        two = cartesianToSpherical(B[:,:3])
        indices = tree.query(two[:,:2], k=1, return_distance=False)

        return np.hstack([B, dist(one, two, indices)])


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


    def boundaryCheck(self, forward, backward, prev):
        clus = []

        if len(backward) == 0:
            return []
        back = np.hstack(backward).T
        one = cartesianToSpherical(back[:,:3])
        tree = BallTree(one[:,:2])

        prevtree = BallTree(prev.points[:,:3])
        
        for cluster_ in forward:
            cluster_ = cluster_.T
            zs = np.sort(cluster_[:,2])
            median_vert = zs[len(zs)//2]

            upper = cluster_[cluster_[:,2] >= median_vert,:]

            cluster = upper

            mean = cluster.mean(axis=0)
            theta = thetaV(cluster[:,:2], mean[:2])
            ang = np.argsort(theta)

            clu = cluster[ang]

            nh = len(clu)//2
            left = clu[:nh+1]
            right = clu[nh-1:]

            # neighbors
            neigh_left = np.hstack(prevtree.query_radius(left, self.boundarycheck_radius))
            neigh_right = np.hstack(prevtree.query_radius(right, self.boundarycheck_radius))

            nln = len(neigh_left)
            nrn = len(neigh_right)

            if nln > 0 and nrn > 0:
                # bad -> filter out
                pass
            elif nln <= 0 and nrn <= 0:
                # good
                clus.append(cluster)
            else:
                two = cartesianToSpherical(cluster_[:,:3])
                indices = tree.query(two[:,:2], k=1, return_distance=False)

                A = np.hstack([cluster_, dist(one, two, indices)])
                
                Af = A[A[:,4] < 0]
                if len(Af) > 0:
                    clus.append(cluster_)
                    
        return clus


    def tracking(self, clusters):
        dim = self.dim
        clus = [c[:dim,:] for c in clusters if c.shape[1] > 4] # more than 4 points
        T = 1

        self.tracker_manager.match(clus, T=T)
        tracks = [track for track in self.tracker_manager.tracks if track.isAlive()]
        
        traj, ext = filterByFrameByFrameVelocity(tracks, self.velocity_threshold)
        
        return traj, None, ext, tracks


    def handle(self, points):
        points = points[points[:,2]<1.82] 
        self.sink.publish("/lidar/all", "pointcloud", points[:,:4], axis="xyzi")
        points = points[:,:4]

        if self.rmground:
            points = self.remove_ground(points)
            self.sink.publish("/lidar/nonGround", "pointcloud", points[:,:3], axis="xyzi")

        curr = PointCloud(points, KDTree(points[:,:3]))
        if self.last is None:
            self.last = curr
            return
        
        
        pointsB, pointsF, f, b = self.difference(curr, self.last, self.threshold)
        self.sink.publish("/dist/backward", "pointcloud", b, axis="xyzid")
        self.sink.publish("/dist/forward", "pointcloud", f, axis="xyzid")

        self.sink.publish("/diff/backward", "pointcloud", pointsB, axis="xyzid")
        self.sink.publish("/diff/forward", "pointcloud", pointsF, axis="xyzid")
        pointsF, clusters = self.clustering(pointsF, self.eps1, self.min_samples1)
        self.sink.publish("/filter/forward", "pointcloud", pointsF, axis=list("xyzc") + ["labels"])
        
        pointsB, clustersB = self.clustering(pointsB, self.eps1, self.min_samples1)
        self.sink.publish("/filter/backward", "pointcloud", pointsB, axis=list("xyzc"))
        
        A = self.boundaryCheck(clusters, clustersB, self.last)

        if len(A) == 0:
            A = np.zeros((0, 3))
        else:
            A_ = [np.hstack([a, np.ones((a.shape[0], 1))*i]) for i, a in enumerate(A)]
            self.sink.publish("/check", "pointcloud", np.vstack(A_), axis="xyzc")
            A = np.vstack(A)
            
        As = self.smooth(curr, A)
        self.sink.publish("/filter/angleF/smooth", "pointcloud", As, axis="xyzid")

        
        points, clusters = self.clustering(As, self.eps2, self.min_samples2)
        self.sink.publish("/dbscan", "pointcloud", points, axis=list("xyzi") + ["labels"])
        
        pos, pre, ext, tracks = self.tracking(clusters)
        self.sink.publish("/tracker/position", "lines", pos, color="red")
        self.sink.publish("/tracker/extend", "ellipse", extendToEllipse([e[0] for e in ext], [e[1] for e in ext]))
        self.last = curr