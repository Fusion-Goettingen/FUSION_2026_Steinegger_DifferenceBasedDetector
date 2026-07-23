import numpy as np, time, json
from .idataloader import DataloaderBase, DataloaderBuilderBase
from pathlib import Path

class AnnotationRLiViT:
    def __init__(self,
                frame: int,
                track_id: int,
                type: str,
                truncated: float,
                occluded: int,
                alpha: float,
                left: int,
                top: int,
                right: int,
                bottom: int,
                height: float,
                width: float,
                length: float,
                x: float,
                y: float,
                z: float,
                rotation: float,
                score: float
                ):
        self.frame = int(frame)
        self.track_id = int(track_id)
        self.type = str(type)
        self.truncated = float(truncated)
        self.occluded = int(occluded)
        self.alpha = float(alpha)
        self.left = float(left)
        self.top = float(top)
        self.right = float(right)
        self.bottom = float(bottom)
        self.height = float(height)
        self.width = float(width)
        self.length = float(length)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.rotation = float(rotation)
        self.score = float(score)


    def __repr__(self):
        vs = ", ".join([f"{k}={v}" for k, v in vars(self).items()])
        return f"{self.__class__.__name__}{{{vs}}}"
        #return json.dumps(vars(self))


class RLiViTDataloaderBuilder(DataloaderBuilderBase):
    def __init__(self):
        self.sequences = None
        self.path = None
    

    def getSequenceRange(self, path):
        files = (path/ "point_clouds").iterdir()
        return [int(file.stem) for file in sorted(files)]

    
    def fromParameters(self, path, sequences, **kwargs):
        path = Path(path)
        
        if sequences is None or len(sequences) == 0:
            sequences = self.getSequenceRange(path)
        
        self.setPathSequence(path, sequences)


    def setPathSequence(self, path, sequences):
        assert path is not None
        assert sequences is not None
        self.sequences = sequences
        
        if sequences is None or len(sequences) == 0:
            self.sequences = self.getSequenceRange(path)

        self.path = path
        return self
    
    def build(self, perf=None):
        dataloader = RLiViTDataloader(**vars(self))
        if perf is not None:
            dataloader.run = perf.wrap(dataloader.run)
        return dataloader


class RLiViTDataloader(DataloaderBase):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


    @staticmethod
    def builder():
        return RLiViTDataloaderBuilder()
    

    def loadAnnotations(self, sequence):
        anno_path = self.path / "labels" / f"{str(sequence).rjust(4, '0')}.txt" 
        
        with open(anno_path, "r") as file:
            return [AnnotationRLiViT(*line.split()) for line in file.readlines()]


    def run(self, handler):
        raise DeprecationWarning()
        lidar_path = self.path/ "point_clouds" / f"{self.sequence}".rjust(4, "0")
        annos = self.loadAnnotations(self.sequence)

        for file in sorted(lidar_path.iterdir()):
            frame = int(file.stem)
            annos_ = [anno for anno in annos if anno.frame == frame]
            handler.sink.publish("/anno/boundingbox", "boundingbox", annos_)
            
            data = np.fromfile(file, dtype=np.float32).reshape((-1, 4))
            #print(data[:12])
            handler.handle(data)
            #time.sleep(1)

            if self.wait:
                cmd = input(f"\r{frame}>")
                match cmd:
                    case "x":
                        break

    
    def iterSequences(self, return_annotations=False):
        for sequence in self.sequences:
            lidar_path = self.path/ "point_clouds" / f"{sequence}".rjust(4, "0")
            annos = self.loadAnnotations(sequence)

            def iterFrames():
                for file in sorted(lidar_path.iterdir()):
                    frame = int(file.stem)
                    annos_ = [anno for anno in annos if anno.frame == frame]
                    data = np.fromfile(file, dtype=np.float32).reshape((-1, 4))

                    if return_annotations:
                        yield annos_, data
                    else:
                        yield data

            yield sequence, iterFrames

    def terminate(self):
        pass


