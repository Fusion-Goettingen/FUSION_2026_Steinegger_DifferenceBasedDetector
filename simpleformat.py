from pathlib import Path
import os, numpy as np, json
from pprint import pprint

class SimpleFormat:
    def __init__(self):
        self.cache = []


    def setOutputDir(self, path):
        self.path = Path(path)
        if not path.is_absolute():
            self.path = Path.cwd() / self.path
        return self
    

    def submit(self, data:list, annos):
        self.cache.append((data, annos))


    def flush(self, seq_id):
        path = Path.cwd() / self.path
        os.makedirs(path, exist_ok=True)
        path = path / (f"{seq_id}".rjust(4, '0') + ".txt")
        
        data = {"seq": seq_id, "results":[]}
        for entry, anno in self.cache:
            tr = [{"pos": e.position.flatten()[:2].tolist(), "ext":e.extend.tolist(), "track_id": e.track_id} for e in entry]
            gt = []
            for a in anno:
                pos = np.array([a.x, a.y])
                w, l, theta = a.width/2, a.length/2, a.rotation
                semi = np.array([[w, 0], [0, l]])
                R = np.array([ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                cov = R.T @ semi @ R
                gt.append({"pos": pos.tolist(), "ext": cov.tolist(), "track_id": a.track_id, "type": a.type})

            data["results"].append({"frame":a.frame, "gt":gt, "tr":tr})
            
        print(f"simpleformat.py::writing to {path}")
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(data, indent=4))
        self.cache = []