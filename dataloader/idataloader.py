class DataloaderBuilderBase:
    def usesRclpy(self):
        return False
    

    def setRclpyNode(self, node):
        pass

    def build(self, *args, **kwargs):
        pass
    

class DataloaderBase:
    def run(self):
        pass


    def iter(self):
        pass
    
    
    @staticmethod
    def builder(handler):
        pass
