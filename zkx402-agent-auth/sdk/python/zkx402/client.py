"""
Synchronous client for zkX402 API
"""

from typing import Dict, List, Any, Optional
import requests
from .exceptions import (
    ZKX402Error,
    PolicyNotFoundError,
    InvalidInputError,
    ProofGenerationError,
    SimulationError,
    NetworkError,
)


class ZKX402Client:
    """
    Synchronous client for zkX402 privacy-preserving authorization.
    
    Example:
        >>> client = ZKX402Client("https://your-server.com")
        >>> client.discover()
        >>> policies = client.list_policies()
        >>> result = client.simulate("simple_threshold", {"amount": 5000, "balance": 10000})
        >>> proof = client.generate_proof("simple_threshold", {"amount": 5000, "balance": 10000})
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize zkX402 client.
        
        Args:
            base_url: Base URL of zkX402 server (e.g., "https://your-server.com")
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._discovery_info: Optional[Dict] = None
        self._policies_cache: Optional[List[Dict]] = None

    def discover(self) -> Dict[str, Any]:
        """
        Discover service capabilities via .well-known/x402 endpoint.
        
        Returns:
            dict: Service metadata including version, capabilities, and endpoints
            
        Raises:
            NetworkError: If discovery request fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/.well-known/x402", timeout=self.timeout
            )
            response.raise_for_status()
            self._discovery_info = response.json()
            return self._discovery_info
        except requests.RequestException as e:
            raise NetworkError(f"Discovery failed: {e}")

    def list_policies(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        List all available authorization policies.
        
        Args:
            refresh: Force refresh cached policies (default: False)
            
        Returns:
            list: List of policy objects with metadata
            
        Raises:
            NetworkError: If request fails
        """
        if self._policies_cache and not refresh:
            return self._policies_cache

        try:
            response = self.session.get(
                f"{self.base_url}/api/policies", timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            self._policies_cache = data.get("policies", [])
            return self._policies_cache
        except requests.RequestException as e:
            raise NetworkError(f"Failed to list policies: {e}")

    def get_policy(self, policy_id: str) -> Dict[str, Any]:
        """
        Get a specific policy by ID.
        
        Args:
            policy_id: Policy identifier (e.g., "simple_threshold")
            
        Returns:
            dict: Policy object
            
        Raises:
            PolicyNotFoundError: If policy doesn't exist
        """
        policies = self.list_policies()
        for policy in policies:
            if policy["id"] == policy_id:
                return policy
        raise PolicyNotFoundError(policy_id)

    def get_policy_schema(self, policy_id: str) -> Dict[str, Any]:
        """
        Get detailed schema for a specific policy.
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            dict: Detailed policy schema with inputs, outputs, examples, and pricing
            
        Raises:
            PolicyNotFoundError: If policy doesn't exist
            NetworkError: If request fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/policies/{policy_id}/schema",
                timeout=self.timeout,
            )
            if response.status_code == 404:
                raise PolicyNotFoundError(policy_id)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if isinstance(e, PolicyNotFoundError):
                raise
            raise NetworkError(f"Failed to get policy schema: {e}")

    def simulate(
        self, policy_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate policy execution without generating zkML proof.
        Fast and free for testing.
        
        Args:
            policy_id: Policy identifier
            inputs: Input values as dict (e.g., {"amount": 5000, "balance": 10000})
            
        Returns:
            dict: Simulation result with approval decision and metadata
            
        Raises:
            PolicyNotFoundError: If policy doesn't exist
            InvalidInputError: If inputs are missing or invalid
            SimulationError: If simulation fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/policies/{policy_id}/simulate",
                json={"inputs": inputs},
                timeout=self.timeout,
            )

            if response.status_code == 404:
                raise PolicyNotFoundError(policy_id)
            elif response.status_code == 400:
                error_data = response.json()
                missing = error_data.get("missing", [])
                raise InvalidInputError(
                    error_data.get("error", "Invalid inputs"), missing
                )

            response.raise_for_status()
            return response.json()
        except (PolicyNotFoundError, InvalidInputError):
            raise
        except requests.RequestException as e:
            raise SimulationError(f"Simulation failed: {e}")

    def generate_proof(
        self, policy_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate zkML proof for authorization policy.
        This creates a verifiable proof that can be used for authorization.
        
        Args:
            policy_id: Policy identifier (used as "model" parameter)
            inputs: Input values as dict
            
        Returns:
            dict: Proof result with proof data, approval status, and verification info
            
        Raises:
            PolicyNotFoundError: If policy doesn't exist
            InvalidInputError: If inputs are missing or invalid
            ProofGenerationError: If proof generation fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate-proof",
                json={"model": policy_id, "inputs": inputs},
                timeout=self.timeout,
            )

            if response.status_code == 404:
                raise PolicyNotFoundError(policy_id)
            elif response.status_code == 400:
                error_data = response.json()
                raise InvalidInputError(error_data.get("error", "Invalid inputs"))

            response.raise_for_status()
            return response.json()
        except (PolicyNotFoundError, InvalidInputError):
            raise
        except requests.RequestException as e:
            raise ProofGenerationError(f"Proof generation failed: {e}")

    def find_policies(
        self,
        category: Optional[str] = None,
        complexity: Optional[str] = None,
        max_proof_time_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find policies matching specified criteria.
        
        Args:
            category: Filter by category (e.g., "financial", "trust", "neural")
            complexity: Filter by complexity ("simple", "medium", "advanced")
            max_proof_time_ms: Maximum acceptable proof time in milliseconds
            
        Returns:
            list: Filtered list of policies
        """
        policies = self.list_policies()
        results = policies

        if category:
            results = [p for p in results if p.get("category") == category]

        if complexity:
            results = [p for p in results if p.get("complexity") == complexity]

        if max_proof_time_ms is not None:
            results = [
                p
                for p in results
                if p.get("avg_proof_time_ms", float("inf")) <= max_proof_time_ms
            ]

        return results

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.session.close()

    def close(self):
        """Close HTTP session"""
        self.session.close()
