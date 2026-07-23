# python3 main.py --profile sink-ros2 load-rlivit -s 130 -W handle-static --rm-ground -t 0.04 -e 1 -m 10 -r 1 -g 4


import numpy as np, sys, pypatchworkpp, scipy, numba#, icp_cpp
from sklearn.cluster import DBSCAN
from .utils import PointCloud, filterByFrameByFrameVelocity#, extendToEllipse

from .utils import extendToEllipse
#from .utils import point_cloud, pathlines, cylinders_markers, ellipse_markers, clustersToEllipses, 
#from scipy.optimize import linear_sum_assignment

#from .utils import clustersToEllipses, covarmat_to_ellipse

from scipy.spatial import KDTree
from sklearn.neighbors import BallTree

from pprint import pprint
from collections import namedtuple


sys.path.append('../random_matrix_tracker')
from random_matrix_tracker import RandomMatrixTracker
sys.path.append('../global_nearest_neighbor')
from global_nearest_neighbor import GlobalNearestNeighborTrackManager


theta = lambda v, w: np.degrees(np.atan2(v[0]*w[1] - v[1]*w[0], v[0]*w[0] + v[1]*w[1]))

@numba.njit(cache=True)
def thetaV(vv, w):
    return np.array([np.arctan2(v[0]*w[1] - v[1]*w[0], v[0]*w[0] + v[1]*w[1]) for v in vv])


def cartesianToSpherical(points):
    depth = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(points[:,1], points[:,0])
    elevation = np.arccos(points[:,2]/ depth)
    return np.vstack([azimuth, elevation, depth]).T


@numba.njit(cache=True)
def freeSpaceCheck(next, prev):
    angle = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    mat = np.zeros((next.shape[0], prev.shape[0]))
    for i, n in enumerate(next):
        for j, p in enumerate(prev):
            mat[i, j] = angle(n, p)
    #scipy.spatial.distance.cdist(next[:,:3], prev[:,:3], angle)
    #print(next.shape, prev.shape, mat.shape)
    min_ang_0 = np.argmin(mat, axis=0)
    min_ang_1 = np.argmin(mat, axis=1)
    return next[min_ang_0], prev[min_ang_1]


@numba.njit(cache=True)
def dist(one, two, indices):
    d = np.empty((len(two)))
    #print(one.shape, two.shape, d.shape, indices.shape)
    for i, (a_, b_) in enumerate(zip(one[indices.ravel()], two)):
        a_ = a_[:3]
        b_ = b_[:3]
        #d_ = (a_@b_)/np.linalg.norm(a_)
        
        #d = ((a_@b_)/np.linalg.norm(a_)**2) * a_
        #d[i] = np.linalg.norm(b_) - d_
        d[i] = np.linalg.norm(b_) - np.linalg.norm(a_)
    return  d[:,None]


class BetaFilter:
    def __init__(self):
        #raise DeprecationWarning("Dont use that")
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
            self.initTracker3D()
        
        elif self.dim == 2:
            self.initTracker2D()
        
        else:
            raise ValueError("dimension needs to be either 2 or 3")

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


    def initTracker3D(self):
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).astype(float)
        Prior = np.eye(6).astype(float)

        Q = np.diag([0.5, 0.5, 0.5, 1, 1, 0.1]).astype(float)
        R = np.eye(3).astype(float) * 0.0005 # precision
        F = lambda T: np.array([[1, 0, 0, T, 0, 0],
                                [0, 1, 0, 0, T, 0],
                                [0, 0, 1, 0, 0, T],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]]).astype(float)

        self.tracker = RandomMatrixTracker(H, Prior, R, F, Q, tau=10, z=0.25)
        self.tracker_manager = GlobalNearestNeighborTrackManager(self.tracker, H, history_size=self.histsize, gate=self.gate)

        x = np.array([[0, 0, 0, 1, 1, 1]]).T.astype(float)
        #gt = np.diag([4, 1])
        X = np.eye(3) * 0.01
        P = np.eye(6).astype(float)
        alpha = 20

        self.tracker.setPrior(x, X, P, alpha)

        return self


    def setThreshold(self, threshold):
        self.threshold = threshold
        return self


    def setQueryRadius(self, radius):
        self.qradius = radius
        return self


    def setK(self, k):
        raise NotImplementedError()
        
    

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
        print(f[:,None].shape, curr.points.shape)
        return last.points[b>threshold], curr.points[f>threshold]#, np.vstack([last.points, b[:,None]])[b>threshold], np.vstack([curr.points, f[:,None]])[f>threshold]


    def angle_difference(self, A, B):
        if len(A) == 0 or len(B) == 0:
            return np.zeros((0, 5))
        
        one = cartesianToSpherical(A[:,:3])
        tree = BallTree(one[:,:2])

        two = cartesianToSpherical(B[:,:3])
        indices = tree.query(two[:,:2], k=1, return_distance=False)

        return np.hstack([B, dist(one, two, indices)])

    def angle_difference_tree(self, A, B):
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
        #smoothB = np.unique(smooth.ravel())
        #print(points.shape, smooth.shape, smoothB.shape)
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


    def filterStationary(self, clusters, last):
        map = icp_cpp.VoxelHashMap3d(0.5, 10, 100)
        map.add(last.points[:,:3])

        filtered = []
        for points in clusters:
            T, corrs, mag = icp_cpp.ICP(points.T, map, 1.0, 3.0, 500, 1e-5)
            #mag1 = np.linalg.norm(T, 'fro')
            mag2 = np.linalg.norm(T[:2,3])# on translation in x, y axis

            if mag2 > 0.01:
                o = np.ones((1, points.shape[1])) * mag2
                pts = np.vstack([points, o])
                filtered.append(pts)
        #print(len(clusters), " -> ",len(filtered))

        points = np.hstack(filtered).T
        return points ,filtered


    def sampling(self, points):
        n = len(points)
        p = np.round(n * 0.1).astype(int)
        idx = np.random.permutation(np.arange(len(points)))
        # print(idx, p)
        return points[idx[:p]]


    """def freeSpaceCheck(self, next, prev):
        angle = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        mat = scipy.spatial.distance.cdist(next[:,:3], prev[:,:3], angle)
        #print(next.shape, prev.shape, mat.shape)
        min_ang = np.argmin(mat, axis=1)
        #pprint(min_ang.shape)"""


    def boundaryCheck(self, forward, backward, prev):
        clus = []
        #print("bC:", len(backward), backward[0].shape)
        #if len(backward) == 0:
        #    #backward = [np.zeros((3, 0))]
        #    return []
        back = np.hstack(backward).T
        #print(back.shape)
        one = cartesianToSpherical(back[:,:3])
        tree = BallTree(one[:,:2])

        #print(forward.shape)
        #print(backward.shape)

        print(prev.points.shape)
        prevtree = BallTree(prev.points[:,:3])
        
        for cluster_ in forward:
            cluster_ = cluster_.T
            #print(cluster_.shape, cluster_)
            zs = np.sort(cluster_[:,2])#.sort()
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
            
            #clus.append((left, right))
            #print(left.shape)
            #print(right.shape)

            # neighbors
            neigh_left = np.hstack(prevtree.query_radius(left, self.boundarycheck_radius))
            neigh_right = np.hstack(prevtree.query_radius(right, self.boundarycheck_radius))

            #print("neigh_L", neigh_left, len(neigh_left))
            #print("neig R", neigh_right, len(neigh_right))
            nln = len(neigh_left)
            nrn = len(neigh_right)

            if nln > 0 and nrn > 0:
                # bad -> filter out
                pass
            elif nln <= 0 and nrn <= 0:
                # good
                clus.append(cluster)
            else:
                #print("consitency check")
                #one = cartesianToSpherical(cluster_[:,:3])
                #tree = BallTree(one[:,:2])

                two = cartesianToSpherical(cluster_[:,:3])
                indices = tree.query(two[:,:2], k=1, return_distance=False)

                A = np.hstack([cluster_, dist(one, two, indices)])
                #print(x.shape)
                Af = A[A[:,4] < 0]
                #print(len(Af), Af.shape)
                if len(Af) > 0:
                    clus.append(cluster_)
                    
        return clus


    def tracking(self, clusters):
        dim = self.dim
        #clus = [c[:self.dim,:] for c in clusters if c.shape[1] > 4] # more than 4 points
        clus = [c[:dim,:] for c in clusters if c.shape[1] > 4] # more than 4 points
        T = 1
        self.tracker_manager.match(clus, T=T)
        tracks = [track for track in self.tracker_manager.tracks if track.isAlive()]
        
        traj, ext = filterByFrameByFrameVelocity(tracks, self.velocity_threshold)

        #pos = [[pose.x.flatten()[:dim] for pose in track.poses] for track in tracks  ]
        #pre = [[track.poses[-1].x.copy().flatten()[:dim], track.poses[-1].predict(1).x.copy().flatten()[:dim]] for track in tracks]
        #e = [[pose.X for pose in track.poses] for track in tracks]
        
        return traj, None, ext, tracks



    def handle(self, points):
        self.sink.publish("/lidar/all", "pointcloud", points[:,:4], axis="xyzi")
        points = points[:,:4]

        if self.rmground:
            points = self.remove_ground(points)
            self.sink.publish("/lidar/nonGround", "pointcloud", points[:,:3], axis="xyzi")

        curr = PointCloud(points, KDTree(points[:,:3]))
        if self.last is None:
            self.last = curr
            return
        
        
        pointsB, pointsF = self.difference(curr, self.last, self.threshold)
        #print(pointsB.shape, pointsF.shape, f.shape, b.shape)
        self.sink.publish("/diff/backward", "pointcloud", pointsB, axis="xyzid")
        self.sink.publish("/diff/forward", "pointcloud", pointsF, axis="xyzid")
        #print(pointsF.shape, pointsB.shape)
        
        pointsF, clusters = self.clustering(pointsF, self.eps1, self.min_samples1)
        self.sink.publish("/filter/forward", "pointcloud", pointsF, axis=list("xyzc") + ["labels"])
        
        pointsB, clustersB = self.clustering(pointsB, self.eps1, self.min_samples1)
        self.sink.publish("/filter/backward", "pointcloud", pointsB, axis=list("xyzc"))
        
        #print(pointsF.shape, pointsB.shape)
        #print(clusters, clustersB)
        
        A = self.boundaryCheck(clusters, clustersB, self.last)
        #A = self.angle_difference(pointsB, pointsF)
        #B = self.angle_difference(pointsF, pointsB)
        #if len(A) > 0:
        #    print("A.shape", A[0].shape)
        #self.sink.publish("/filter/angleF", "pointcloud", np.vstack(A), axis="xyzid")
        #self.sink.publish("/filter/angleB", "pointcloud", B, axis="xyzid")
        
        
        #self.sink.publish("/filter/angleF/threshold", "pointcloud", Af, axis="xyzid")
        #print("A", A)
        if len(A) == 0:
            #print("empty")
            A = np.zeros((0, 3))
        else:
            A = np.vstack(A)
        #print("A", A.shape)
        #print(A.shap.shapee)
        As = self.smooth(curr, A)
        #print("As.shape", As.shape)
        self.sink.publish("/filter/angleF/smooth", "pointcloud", As, axis="xyzid")

        
        points, clusters = self.clustering(As, self.eps2, self.min_samples2)
        self.sink.publish("/dbscan", "pointcloud", points, axis=list("xyzi") + ["labels"])
        #self.sink.publish("/cluster/bounds", "ellipse", clustersToEllipses(clusters, self.dim))
        
        
        #self.sink.publish("/filtered/bounds", "ellipse", clustersToEllipses(clusters, self.dim))
        pos, pre, ext, tracks = self.tracking(clusters)
        #pprint([ vars(track) for track in tracks])
        self.sink.publish("/tracker/position", "lines", pos, color="red")
        self.sink.publish("/tracker/extend", "ellipse", extendToEllipse([e[0] for e in ext], [e[1] for e in ext]))
        self.last = curr