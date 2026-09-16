"""Separate failed experiments from service/configuration errors that require recovery."""


class ExperimentFailed(RuntimeError):
    """This experiment is invalid; record rejection and let research continue."""


class ServiceError(RuntimeError):
    """An API/service failed; preserve the checkpoint and stop, without fallback."""
