"""
Asynchronous client for zkX402 API
"""

from typing import Dict, List, Any, Optional
import aiohttp
from .exceptions import (
    ZKX402Error,
    PolicyNotFoundError,
    InvalidInputError,
    ProofGenerationError,
    SimulationError,
    NetworkError,
)


class AsyncZKX402Client:
    """
    Asynchronous client for zkX402 privacy-preserving authorization.
    
    Example:
        >>> async with AsyncZKX402Client("https://your-server.com") as client:
        >>>     await client.discover()
        >>>     policies = await client.list_policies()
        >>>     result = await client.simulate("simple_threshold", {"amount": 5000, "balance": 10000})
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize async zkX402 client.
        
        Args:
            base_url: Base URL of zkX402 server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._discovery_info: Optional[Dict] = None
        self._policies_cache: Optional[List[Dict]] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        return self._session

    async def discover(self) -> Dict[str, Any]:
        """Discover service capabilities"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/.well-known/x402") as resp:
                resp.raise_for_status()
                self._discovery_info = await resp.json()
                return self._discovery_info
        except aiohttp.ClientError as e:
            raise NetworkError(f"Discovery failed: {e}")

    async def list_policies(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """List all available policies"""
        if self._policies_cache and not refresh:
            return self._policies_cache

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/policies") as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._policies_cache = data.get("policies", [])
                return self._policies_cache
        except aiohttp.ClientError as e:
            raise NetworkError(f"Failed to list policies: {e}")

    async def get_policy(self, policy_id: str) -> Dict[str, Any]:
        """Get a specific policy by ID"""
        policies = await self.list_policies()
        for policy in policies:
            if policy["id"] == policy_id:
                return policy
        raise PolicyNotFoundError(policy_id)

    async def get_policy_schema(self, policy_id: str) -> Dict[str, Any]:
        """Get detailed schema for a policy"""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/policies/{policy_id}/schema"
            ) as resp:
                if resp.status == 404:
                    raise PolicyNotFoundError(policy_id)
                resp.raise_for_status()
                return await resp.json()
        except PolicyNotFoundError:
            raise
        except aiohttp.ClientError as e:
            raise NetworkError(f"Failed to get policy schema: {e}")

    async def simulate(
        self, policy_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate policy execution without zkML proof"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/policies/{policy_id}/simulate",
                json={"inputs": inputs},
            ) as resp:
                if resp.status == 404:
                    raise PolicyNotFoundError(policy_id)
                elif resp.status == 400:
                    error_data = await resp.json()
                    missing = error_data.get("missing", [])
                    raise InvalidInputError(
                        error_data.get("error", "Invalid inputs"), missing
                    )
                resp.raise_for_status()
                return await resp.json()
        except (PolicyNotFoundError, InvalidInputError):
            raise
        except aiohttp.ClientError as e:
            raise SimulationError(f"Simulation failed: {e}")

    async def generate_proof(
        self, policy_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate zkML proof for authorization"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/generate-proof",
                json={"model": policy_id, "inputs": inputs},
            ) as resp:
                if resp.status == 404:
                    raise PolicyNotFoundError(policy_id)
                elif resp.status == 400:
                    error_data = await resp.json()
                    raise InvalidInputError(error_data.get("error", "Invalid inputs"))
                resp.raise_for_status()
                return await resp.json()
        except (PolicyNotFoundError, InvalidInputError):
            raise
        except aiohttp.ClientError as e:
            raise ProofGenerationError(f"Proof generation failed: {e}")

    async def find_policies(
        self,
        category: Optional[str] = None,
        complexity: Optional[str] = None,
        max_proof_time_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find policies matching criteria"""
        policies = await self.list_policies()
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

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
