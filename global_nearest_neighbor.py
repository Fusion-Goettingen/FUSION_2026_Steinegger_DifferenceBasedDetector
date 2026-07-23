#!/usr/bin/env python
# coding: utf-8

import numpy as np, copy
from scipy.optimize import linear_sum_assignment
from gauss_wasserstein import gauss_wasserstein_distance_cho as gauss_wasserstein_distance
from pprint import pprint
from sqrtm import sqrtmvec
from collections import namedtuple

from enum import Enum

np.set_printoptions(suppress=True)

eucl_sq = lambda a, b: np.linalg.norm(a - b, axis=0) ** 2

def prior(meas):
    #x = np.stack([np.mean(meas, axis=1), np.zeros(2)])
    x = np.hstack([np.mean(meas, axis=1), np.zeros(2)])[:,None]
    X = np.cov(meas)

    return x, X


class State(Enum):
    INIT = 0
    NEW = 1
    TRACKED = 2
    LOST = 3
    TERMINATED = -1

class Observation(Enum):
    DETECTED = 0
    MISSED = 1


def transition(state, cnt, obs):
    #print(state, cnt, obs)
    match (state, cnt, obs):
        case (State.INIT, _, _):
            new_state = (State.NEW, 0)
        
        case (State.NEW, 5, Observation.DETECTED):
            new_state = (State.TRACKED, 0)
        case (State.NEW, _, Observation.DETECTED):
            new_state = (State.NEW, cnt+1)
        case (State.NEW, _, Observation.MISSED):
            new_state = (State.TERMINATED, -1)
        
        case (State.TRACKED, _, Observation.DETECTED):
            new_state = (State.TRACKED, cnt)
        case (State.TRACKED, _, Observation.MISSED):
            new_state = (State.LOST, 0)
        
        case (State.LOST, 5, Observation.MISSED):
            new_state = (State.TERMINATED, -1)
        case (State.LOST, _, Observation.DETECTED):
            new_state = (State.TRACKED, 0)
        case (State.LOST, _, Observation.MISSED):
            new_state = (State.LOST, cnt+1)
        
        case _:
            new_state = (State.TERMINATED, -1)
    #s, c = new_state
    #print(s, c)
    #print()
    return new_state


class Track:
    counter = 0
    def __init__(self, t_start, state, history_size, points):
        self.poses = [copy.deepcopy(state)]
        #self.poses = [state]
        self.t_start = t_start
        self.t_ = t_start
        self.id = Track.counter
        self.history_size = history_size
        self.points = [points]
        self.covs = [np.cov(points)]
        Track.counter += 1

        self.track_state = (State.INIT, 0)

        self.heights = [np.min(points[:,2]), np.max(points[:,2])]


    def add(self, x, points=None):
        #print("Track.add:x", x)
        self.poses.append(x)
        self.points.append(points)
        self.covs.append(np.cov(points) if points is not None else self.covs[-1])

        self.heights.append([np.min(points[:,2]), np.max(points[:,2])] if points is not None else (None, None))

        self.t_ += 1

        if self.history_size is not None:
            self.poses = self.poses[-self.history_size:]
        
        if points is None:
            self.track_state = transition(*self.track_state, Observation.MISSED)
        else:
            self.track_state = transition(*self.track_state, Observation.DETECTED)


    def dumps(self):
        repr = self.__dict__
        #print("Track.dumps", repr["poses"][0].dumps())
        repr["poses"] = [pose.dumps() for pose in repr["poses"]]
        repr["points"] = [points.tolist() for points in repr["points"]]
        repr["covs"] = [points.tolist() for points in repr["covs"]]
        repr["heights"] = [height for height in repr["heights"]]
        #repr[""]
        return repr
    

    def isAlive(self):
        return self.track_state[0] != State.TERMINATED
    

    def isActive(self):
        return self.track_state[0] in [State.TRACKED, State.NEW]
    

    


class GlobalNearestNeighborTrackManager:
    def __init__(self, tracker, H, history_size, gate=None):#, distance=eucl_sq):
        self.gate = gate
        self.tracks = None
        self.t = 0
        self.tracker = tracker
        self.H = H
        self.history_size = history_size
        
        #self.distance = distance

    
    def cost_matrix(self, detec, tracks):
        last_poses = np.array([track.poses[-1].x for track in tracks])
        #print("costmatrix", tracks[0][1].dumps())
        prev_points = [track.covs[-1] for track in tracks]
        now_means = np.array([meas.mean(axis=1) for meas in detec])
        prev_means = np.hstack(np.array([self.H @ pose for pose in last_poses])).T

        #@numba.njit(cache=True)
        def cm(detec, prev_means, prev_points, now_means, gate):
            
            now_covs = [np.cov(meas) for meas in detec]
            #now_cov_roots = [scipy.linalg.sqrtm(cov) for cov in now_covs]
            now_cov_roots = sqrtmvec(now_covs)
            
            mat = np.zeros((len(now_means), len(prev_means)))
            
            for i, (nm, nC, nCr) in enumerate(zip(now_means, now_covs, now_cov_roots)):
                for j, (pm, pC) in enumerate(zip(prev_means, prev_points)):
                    #v = np.linalg.norm(nm - pm, axis=0) ** 2
                    try:
                        v = gauss_wasserstein_distance((nm, nC), (pm, pC), nCr)
                    except np.linalg.LinAlgError as e:
                        print((nm, nC))
                        print((pm, pC))
                        print(nCr)
                        raise e
                    
                    mat[i, j] = v
                    if gate is not None and np.sqrt(v) > gate:
                        mat[i, j] = 1e9

            # Dummy nodes
            C = mat
            if gate is not None:
                x, y = np.array(mat.shape) * 2
                
                C = np.ones((x, y)) * (gate ** 2)
                C[:mat.shape[0], :mat.shape[1]] = mat

            return C, now_means, prev_means
        return cm(detec, prev_means, prev_points, now_means, self.gate)
    


    def match(self, detec, T):
        track_init = lambda t, x: Track(t, self.tracker.prior(*prior(x)), history_size=self.history_size, points=x)
        if self.tracks is None or len(self.tracks) <= 0:
            self.tracks = [track_init(self.t, y) for y in detec]
            return
        self.t += 1

        #tracks = [(i, track) for i, track in enumerate(self.tracks) if self.t-5 < track.t_ <= self.t]
        tracks = [track for track in self.tracks if track.isAlive()]
        
        if len(detec) == 0:
            # TODO check
            for i, track in enumerate(tracks):
                tracks[i].add(tracks[i].poses[-1].predict(T))
                #print("uh")
            return

        C, now_means, prev_means = self.cost_matrix(detec, tracks)
        prv, nxt = linear_sum_assignment(C.T) # row, col = ...

        unmatched = []
        for i, j in zip(prv, nxt):
            #if i >= prev_means.shape[0] or j >= now_means.shape[0]:
            #    unmatched.append((i, j))
            #    continue
            if i >= prev_means.shape[0] and j >= now_means.shape[0]:
                # dummy
                continue
                
            elif i >= prev_means.shape[0]:
                track_ = track_init(self.t, detec[j].copy())
                self.tracks.append(track_)

            elif j >= now_means.shape[0]:
                tracks[i].add(tracks[i].poses[-1].predict(T))
            else:
                dd = detec[j]
                track = tracks[i]
                state = track.poses[-1]
                track.add(state.update(dd).predict(T), points=dd)
        return prv, nxt


    def dumps(self):
        repr = [track.dumps() for track in self.tracks]
        return repr
    

    def getCurrentState(self):
        if self.tracks is None:
            return []#[Record(None, None, None, None)]

        tracks = self.tracks

        t_latest = max([track.t_ for track in tracks] + [-1])

        entries = []
        for id, track in enumerate(tracks):
            if track.t_ != t_latest:
                continue
            pose = track.poses[-1]
            entry = Record(track.t_, id, pose.x, pose.X, track.heights[0], track.heights[1], track.track_state)
            #print(entry)
            entries.append(entry)
        return entries

Record = namedtuple("Record", ["frame", "track_id", "position", "extend", "z_min", "z_max", "track_state"])