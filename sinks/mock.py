from .isink import SinkBase

class MockSink(SinkBase):
    def __init__(self):
        pass
    

    def fromParameters(self, **kwargs):
        return self
    

    def setOutputDir(self, outdir):
        return self
    

    def publish(self, name, type_, data, *args, **kwargs):
        pass