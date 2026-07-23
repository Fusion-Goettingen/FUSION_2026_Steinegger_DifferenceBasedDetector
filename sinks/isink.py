class SinkBase:
    def usesRclpy(self):
        return False
    

    def publish(self, name, type_, data, *args, **kwargs):
        pass