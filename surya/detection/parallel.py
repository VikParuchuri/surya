class FakeFuture:
    def __init__(self, func, *args, **kwargs):
        self._result = func(*args, **kwargs)

    def result(self):
        return self._result

class FakeExecutor:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *excinfo):
        pass

    def submit(self, fn, *args, **kwargs):
        return FakeFuture(fn, *args, **kwargs)
