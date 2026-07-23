from .isink import SinkBase
from pathlib import Path
from collections import defaultdict
import json, os

from pprint import pprint

class FileSink(SinkBase):
    def __init__(self):
        self.cnt = defaultdict(lambda:0)
        self.outdir = None
    

    def fromParameters(self, **kwargs):
        self.setOutputDir(kwargs["outdir"])
        return self
    

    def setOutputDir(self, outdir):
        self.outdir = Path(outdir)
        return self
    

    def publish(self, name, type_, data, *args, **kwargs):
        if name[0] == "/":
            name = name[1:]
        cnt = self.cnt[name]
        
        path = Path.cwd() / self.outdir / Path(name) #/ f"{cnt:04}.json"
        
        #path.touch(exist_ok=True)
        os.makedirs(path, exist_ok=True)
        with open(path / f"{cnt:04}.json", "w") as file:
            match type_:
                case "pointcloud":
                    file.write(json.dumps([x.tolist() for x in data]))
                case "ellipse":
                    #pprint(data)
                    d = [{k:v.tolist() if k == "xyz" else v for k,v in x._asdict().items()} for x in data]
                    #pprint(d)
                    file.write(json.dumps(d))
                case "line":
                    json.dump(file, [x._asdict() for x in data])
        self.cnt[name] += 1