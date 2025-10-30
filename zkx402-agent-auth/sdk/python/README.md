# zkX402 Python SDK

Privacy-preserving authorization for AI agents using zkML proofs.

## Installation

```bash
pip install zkx402
```

## Quick Start

### Synchronous Client

```python
from zkx402 import ZKX402Client

# Initialize client
client = ZKX402Client("http://localhost:3001")

# Discover service capabilities
discovery = client.discover()
print(f"Service: {discovery['service']}")
print(f"Total policies: {discovery['total_policies']}")

# List all policies
policies = client.list_policies()
for policy in policies:
    print(f"- {policy['name']}: {policy['description']}")

# Simulate a policy (fast, free testing)
result = client.simulate("simple_threshold", {
    "amount": 5000,
    "balance": 10000
})
print(f"Simulation result: {result['approved']}")  # True

# Generate a verifiable zkML proof
proof = client.generate_proof("simple_threshold", {
    "amount": 5000,
    "balance": 10000
})
print(f"Proof generated: {proof['proof'][:32]}...")
print(f"Verified: {proof['verification']['verified']}")

client.close()
```

### Async Client

```python
import asyncio
from zkx402 import AsyncZKX402Client

async def main():
    async with AsyncZKX402Client("http://localhost:3001") as client:
        # List policies
        policies = await client.list_policies()
        
        # Simulate policy
        result = await client.simulate("simple_threshold", {
            "amount": 5000,
            "balance": 10000
        })
        print(f"Approved: {result['approved']}")
        
        # Generate proof
        proof = await client.generate_proof("simple_threshold", {
            "amount": 5000,
            "balance": 10000
        })
        print(f"Proof verified: {proof['verification']['verified']}")

asyncio.run(main())
```

## Features

- **Service Discovery** - Automatic endpoint and capability detection
- **Policy Browsing** - List and filter available authorization policies
- **Fast Simulation** - Test policies instantly without generating proofs
- **Proof Generation** - Create verifiable zkML proofs for authorization
- **Type Hints** - Full type annotations for better IDE support
- **Async Support** - Both sync and async clients available
- **Error Handling** - Custom exceptions for better error management

## Advanced Usage

### Finding Policies

```python
# Find fast policies (< 2s proof time)
fast_policies = client.find_policies(max_proof_time_ms=2000)

# Find financial policies
financial = client.find_policies(category="financial")

# Find simple complexity policies
simple = client.find_policies(complexity="simple")
```

### Getting Policy Details

```python
# Get policy schema
schema = client.get_policy_schema("simple_threshold")
print(f"Inputs: {schema['schema']['inputs']}")
print(f"Pricing: {schema['pricing']}")
print(f"Examples: {schema['examples']}")
```

### Context Manager

```python
with ZKX402Client("http://localhost:3001") as client:
    result = client.simulate("simple_threshold", {"amount": 5000, "balance": 10000})
    print(result)
# Client automatically closed
```

## Error Handling

```python
from zkx402 import (
    PolicyNotFoundError,
    InvalidInputError,
    ProofGenerationError,
    SimulationError,
)

try:
    proof = client.generate_proof("invalid_policy", {"amount": 5000})
except PolicyNotFoundError as e:
    print(f"Policy not found: {e.policy_id}")
except InvalidInputError as e:
    print(f"Missing inputs: {e.missing_inputs}")
except ProofGenerationError as e:
    print(f"Proof failed: {e}")
```

## API Reference

### ZKX402Client

#### Methods

- `discover()` - Discover service capabilities
- `list_policies(refresh=False)` - List all policies
- `get_policy(policy_id)` - Get specific policy
- `get_policy_schema(policy_id)` - Get detailed policy schema
- `simulate(policy_id, inputs)` - Simulate policy without proof
- `generate_proof(policy_id, inputs)` - Generate zkML proof
- `find_policies(category, complexity, max_proof_time_ms)` - Find policies matching criteria

### AsyncZKX402Client

Same methods as ZKX402Client but async (use `await`).

## Examples

See the [examples directory](./examples/) for more detailed examples:

- `basic.py` - Basic usage
- `async_example.py` - Async usage
- `policy_discovery.py` - Discovering and filtering policies
- `error_handling.py` - Comprehensive error handling

## Documentation

- [Agent Integration Guide](../../AGENT_INTEGRATION.md)
- [API Reference](../../API_REFERENCE.md)
- [zkX402 Documentation](https://github.com/hshadab/zkx402)

## License

MIT
