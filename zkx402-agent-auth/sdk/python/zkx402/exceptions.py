"""Custom exceptions for zkX402 SDK"""


class ZKX402Error(Exception):
    """Base exception for zkX402 SDK"""

    pass


class PolicyNotFoundError(ZKX402Error):
    """Raised when a policy ID does not exist"""

    def __init__(self, policy_id: str):
        self.policy_id = policy_id
        super().__init__(f"Policy not found: {policy_id}")


class InvalidInputError(ZKX402Error):
    """Raised when inputs are missing or invalid"""

    def __init__(self, message: str, missing_inputs=None):
        self.missing_inputs = missing_inputs or []
        super().__init__(message)


class ProofGenerationError(ZKX402Error):
    """Raised when proof generation fails"""

    pass


class SimulationError(ZKX402Error):
    """Raised when policy simulation fails"""

    pass


class NetworkError(ZKX402Error):
    """Raised when network request fails"""

    pass
