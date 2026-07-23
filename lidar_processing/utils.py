from collections import namedtuple
import numpy as np
from pprint import pprint

#Ellipse = namedtuple("Ellipse", ["xyz", "eigval", "psi", "dim"])

PointCloud = namedtuple("PointCloud", ["points", "tree"])
Ellipse = namedtuple("Ellipse", ["xyz", "eigval", "psi", "dim"])

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


def filterByFrameByFrameVelocityv1(tracks, velocity_threshold):
    #print(tracks)
    
    
    traj = []
    for track in tracks:
        old = track.poses[0]
        tt = [old.x[:2].flatten()]
        for pos in track.poses[1:]:
            if np.linalg.norm(np.array(pos.x) - np.array(old.x)) > velocity_threshold:
                tt.append(pos.x[:2].flatten())
            else:
                #t.append(tt)
                traj.append(tt)
                tt = [pos.x[:2].flatten()]
            old = pos
        traj.append(tt)
        #traj.append(t)
    
    #pprint(traj)
    # AHHHHHH
    flat = set([(p[0], p[1]) for t in traj for p in t])
    #print(flat)
    def foo(nd):
        #print(nd)
        nd = nd.flatten()
        x, y = nd[0], nd[1]
        #print((x, y))
        return(x, y)
    ext = [(track.poses[-1].x, track.poses[-1].X) for track in tracks if foo(track.poses[-1].x) in flat]
    return traj, ext


def filterByFrameByFrameVelocity(tracks, velocity_threshold):
    traj = []
    for track in tracks:
        old = track.poses[0]
        tt = [old]
        for pos in track.poses[1:]:
            if np.linalg.norm(np.array(pos.x) - np.array(old.x)) > velocity_threshold:
                tt.append(pos)
            else:
                tt = [pos]
            old = pos
        traj.append(tt)
    
    pos = [[p.x[:2].flatten() for p in t] for t in traj]
    ext = [[t[-1].x[:2].flatten(), t[-1].X] for t in traj if len(t) >= 2]
    #print(pos)
    #print(ext)
    return pos, ext



def extendToEllipse(pos, ext):
    #print(pos)
    #print(ext)
    ell = []
    for m, C in zip(pos, ext):
        m = m[:2]
        eig_vals, eig_vecs = np.linalg.eig(C)
        psi = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
        eigval = [3*np.sqrt(ev) for ev in eig_vals] # *2 -> 95%
        c = Ellipse(m, eigval, psi, 2)
        ell.append(c)
        #print(c)
    return ell