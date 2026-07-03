"""Custom exceptions for waterkit."""


class WaterKitError(Exception):
    """Base exception for all waterkit errors."""

    pass


class ValidationError(WaterKitError):
    """Raised when input validation fails."""

    pass


class MapError(WaterKitError):
    """Raised when an AutoDock map cannot be read, generated, or interpolated."""

    pass


class PerceptionError(WaterKitError):
    """Raised when hydrogen-bond or rotatable-bond perception fails."""

    pass


class ForceFieldError(WaterKitError):
    """Raised when force-field parameters are missing or inconsistent."""

    pass
