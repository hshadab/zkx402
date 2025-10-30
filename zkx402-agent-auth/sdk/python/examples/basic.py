"""Basic usage example for zkX402 Python SDK"""

import sys
sys.path.insert(0, '..')

from zkx402 import ZKX402Client

def main():
    # Initialize client
    client = ZKX402Client("http://localhost:3001")
    
    print("=== zkX402 Python SDK Example ===\n")
    
    # Discover service
    print("1. Discovering service...")
    discovery = client.discover()
    print(f"   Service: {discovery['service']}")
    print(f"   Version: {discovery['version']}")
    print(f"   Total policies: {len(discovery.get('pre_built_policies', []))}\n")
    
    # List policies
    print("2. Listing policies...")
    policies = client.list_policies()
    for p in policies[:4]:  # Show first 4
        print(f"   - {p['name']}: {p['description']}")
    print(f"   ... and {len(policies) - 4} more\n")
    
    # Simulate a policy
    print("3. Simulating simple_threshold policy...")
    sim_result = client.simulate("simple_threshold", {
        "amount": 5000,
        "balance": 10000
    })
    print(f"   Approved: {sim_result['approved']}")
    print(f"   Execution time: {sim_result['execution_time_ms']}")
    print(f"   Estimated proof time: {sim_result['proof_generation']['estimated_time']}\n")
    
    # Find fast policies
    print("4. Finding fast policies (< 2s)...")
    fast = client.find_policies(max_proof_time_ms=2000)
    print(f"   Found {len(fast)} fast policies\n")
    
    print("=== Example Complete ===")
    
    client.close()

if __name__ == "__main__":
    main()
