from concurrent.futures import Future, Executor

class FakeFuture(Future):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        try:
            self.set_result(func(*args, **kwargs))
        except Exception as e:
            self.set_exception(e)

class FakeExecutor(Executor):
    def __init__(self, max_workers=None):
        super().__init__()

    def submit(self, fn, *args, **kwargs):
        return FakeFuture(fn, *args, **kwargs)
