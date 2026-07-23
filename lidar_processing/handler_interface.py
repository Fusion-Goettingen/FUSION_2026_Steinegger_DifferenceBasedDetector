from typing import Protocol


class HandlerInterface(Protocol):
    def setParentNode(self, node):
        raise NotImplementedError("This is the interface class, this functionality is not implemented here")

    
    def handle(self, curr, last, *args, **kwargs):
        raise NotImplementedError("This is the interface class, this functionality is not implemented here.")


