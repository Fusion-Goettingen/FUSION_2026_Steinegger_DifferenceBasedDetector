import json
#from analysis import HandlerPerformanceAnalysis

import copy, datetime

#import rclpy
#from rclpy.node import Node
#from dataloader import ROS2Dataloader
from simpleformat import SimpleFormat
import numpy as np

class ControllerBuilder:
    curr_time = datetime.datetime.now().timetuple()[:6]

    def __init__(self):
        self.last = None
        self.stamp = None
        self.perf = None
        self.handler = None
        self.dump_tracks = False
        self.sink = None
        self.name = None

        self.velo_threshold = 0


    def setName(self, name):
        assert name != None
        self.name = name
        return self
    

    def setVelocityThresholdVisually(self, th):
        #raise DeprecationWarning("to be removed")
        self.velo_threshold = th
        return self


    def setHandler(self, handler):
        self.handler = handler
        return self
    

    def setDataloader(self, dataloader):
        self.dataloader = dataloader
        return self
    

    def setSink(self, sink):
        self.sink = sink
        return self
    

    def setResultsHandlerPath(self, path, run_nr=None):
        dt = ControllerBuilder.curr_time
        self.results_path = path / "".join([f"{e}".rjust(2, '0') for e in dt])
        if run_nr is not None:
            self.results_path = self.results_path / self.name
        print(self.results_path)
        return self


    def build(self):
        handler = self.handler
        dataloader = self.dataloader
        sink = self.sink

        if dataloader.usesRclpy() or sink.usesRclpy():
            if dataloader.usesRclpy() or sink.usesRclpy():
                rclpy.init(args=None)

            node = Node("LiDARPedestrianTracker")
            
            if dataloader.usesRclpy():
                dataloader.setRclpyNode(node)
            if sink.usesRclpy():
                sink.setRclpyNode(node)
        
        handler.velocity_threshold = self.velo_threshold
        handler.setSink(sink)
        controller = Controller(dataloader.build(self.perf), handler, self.results_path)
        
        controller.dump_tracks = self.dump_tracks
        controller.perf = self.perf
        return controller


class Controller:
    def __init__(self, dataloader, handler, results_path):
        self.dataloader = dataloader
        self.handler = handler
        self.last = None
        self.resultHandler = SimpleFormat().setOutputDir(results_path)


    def filterVelocity(self, annos):
        if self.last is None:
            self.last = annos
            return annos
        
        last_ids = set([a.track_id for a in self.last])
        curr_ids = set([a.track_id for a in annos])

        ann = []
        for id in last_ids.intersection(curr_ids):
            last = [a for a in self.last if a.track_id == id][0]
            l = np.array([last.x, last.y])
            curr = [a for a in annos if a.track_id == id][0]
            c = np.array([curr.x, curr.y])

            if np.linalg.norm(l - c) > self.handler.velocity_threshold:
                ann.append(curr)

        self.last = annos
        return ann
    

    def run(self):
        #if isinstance(self.dataloader, ROS2Dataloader):
        #    self.dataloader.run(self.handler.finish())
        #else:
        for seq, frames in self.dataloader.iterSequences(return_annotations=True):  
            handler = copy.copy(self.handler).finish()
            for fnr, (annos, data) in enumerate(frames()):
                print("controller.py::", type(handler).__name__, seq, fnr)
                handler.handle(data)
                if annos is not None:
                    handler.sink.publish("/anno/boundingbox", "boundingbox", self.filterVelocity(annos))
                self.resultHandler.submit(handler.tracker_manager.getCurrentState(), annos)
            self.resultHandler.flush(seq)

    def terminate(self):
        print("terminate")
        if self.dump_tracks:
            path = "./dump_tracks.json"
            with open(path, "w") as file:
                s = self.handler.tracker_manager.dumps()
                json.dump(s, file)
                print("tracks saved to", path)

        self.dataloader.terminate()