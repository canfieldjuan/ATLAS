"""
Custom exceptions for storage layer.

Provides explicit error types instead of silent failures.
"""


class StorageError(Exception):
    """Base exception for all storage errors."""

    pass


class DatabaseUnavailableError(StorageError):
    """Raised when database connection is not available."""

    def __init__(self, operation: str = "database operation"):
        self.operation = operation
        super().__init__(
            f"Database not available for {operation}. "
            "Check database connection and initialization."
        )


class DatabaseOperationError(StorageError):
    """Raised when a database operation fails."""

    def __init__(self, operation: str, cause: Exception):
        self.operation = operation
        self.cause = cause
        super().__init__(f"Database operation '{operation}' failed: {cause}")


class ReminderNotFoundError(StorageError):
    """Raised when a reminder is not found."""

    def __init__(self, reminder_id):
        self.reminder_id = reminder_id
        super().__init__(f"Reminder not found: {reminder_id}")
