class FakeParallel():
    def __init__(self, func, *args):
        self._result = func(*args)

    def result(self):
        return self._result
