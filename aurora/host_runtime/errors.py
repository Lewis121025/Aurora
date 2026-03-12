class AuroraRuntimeError(RuntimeError):
    pass


class HostEnvironmentError(AuroraRuntimeError):
    pass


class CollapseProviderError(AuroraRuntimeError):
    pass
