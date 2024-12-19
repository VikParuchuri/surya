class FakeFuture:
    def __init__(self, func, *args, **kwargs):
        # 初始化 FakeFuture 对象，执行传入的函数并存储结果
        self._result = func(*args, **kwargs)

    def result(self):
        # 返回函数执行结果
        return self._result

class FakeExecutor:
    def __init__(self, **kwargs):
        # 初始化 FakeExecutor 对象
        pass

    def __enter__(self):
        # 进入上下文管理器时返回自身
        return self

    def __exit__(self, *excinfo):
        # 退出上下文管理器时不做任何操作
        pass

    def submit(self, fn, *args, **kwargs):
        # 提交任务，返回 FakeFuture 对象
        return FakeFuture(fn, *args, **kwargs)
