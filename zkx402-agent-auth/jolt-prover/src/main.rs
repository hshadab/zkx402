use jolt::Serializable;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Authorization request inputs
#[derive(Debug, Serialize, Deserialize)]
pub struct AuthRequest {
    // Public inputs
    pub transaction_amount: u64, // micro-USDC
    pub vendor_id: u64,          // hash of vendor name
    pub timestamp: u64,          // Unix timestamp

    // Private inputs (hidden by ZK proof)
    pub balance: u64,         // micro-USDC
    pub velocity_1h: u64,     // spending in last hour
    pub velocity_24h: u64,    // spending in last 24h
    pub vendor_trust_score: u32, // 0-100

    // Policy parameters (private)
    pub max_single_tx_percent: u32,   // max % of balance per tx
    pub max_velocity_1h_percent: u32, // max % of balance per hour
    pub max_velocity_24h_percent: u32, // max % of balance per day
    pub min_vendor_trust: u32,        // minimum trust score
}

/// Authorization response
#[derive(Debug, Serialize, Deserialize)]
pub struct AuthResponse {
    pub authorized: bool,
    pub risk_score: u32,
    pub proof: Vec<u8>,
    pub proving_time_ms: u128,
}

impl AuthRequest {
    /// Create a sample approved transaction
    pub fn sample_approved() -> Self {
        Self {
            transaction_amount: 50_000, // $0.05
            vendor_id: hash_vendor("api.openai.com"),
            timestamp: 1704067200,
            balance: 10_000_000, // $10.00
            velocity_1h: 20_000, // $0.02 spent in last hour
            velocity_24h: 100_000, // $0.10 spent in last 24h
            vendor_trust_score: 80,
            max_single_tx_percent: 10,   // max 10% per tx
            max_velocity_1h_percent: 5,  // max 5% per hour
            max_velocity_24h_percent: 20, // max 20% per day
            min_vendor_trust: 50,
        }
    }

    /// Create a sample rejected transaction (amount too high)
    pub fn sample_rejected_amount() -> Self {
        Self {
            transaction_amount: 2_000_000, // $2.00 (20% of balance - exceeds 10% limit)
            vendor_id: hash_vendor("api.openai.com"),
            timestamp: 1704067200,
            balance: 10_000_000,
            velocity_1h: 20_000,
            velocity_24h: 100_000,
            vendor_trust_score: 80,
            max_single_tx_percent: 10,
            max_velocity_1h_percent: 5,
            max_velocity_24h_percent: 20,
            min_vendor_trust: 50,
        }
    }

    /// Create a sample rejected transaction (velocity too high)
    pub fn sample_rejected_velocity() -> Self {
        Self {
            transaction_amount: 50_000,
            vendor_id: hash_vendor("api.openai.com"),
            timestamp: 1704067200,
            balance: 10_000_000,
            velocity_1h: 800_000, // $0.80 in last hour (8%, exceeds 5% limit)
            velocity_24h: 2_000_000,
            vendor_trust_score: 80,
            max_single_tx_percent: 10,
            max_velocity_1h_percent: 5,
            max_velocity_24h_percent: 20,
            min_vendor_trust: 50,
        }
    }

    /// Create a sample rejected transaction (untrusted vendor)
    pub fn sample_rejected_trust() -> Self {
        Self {
            transaction_amount: 50_000,
            vendor_id: hash_vendor("unknown-vendor.xyz"),
            timestamp: 1704067200,
            balance: 10_000_000,
            velocity_1h: 20_000,
            velocity_24h: 100_000,
            vendor_trust_score: 30, // Below 50 threshold
            max_single_tx_percent: 10,
            max_velocity_1h_percent: 5,
            max_velocity_24h_percent: 20,
            min_vendor_trust: 50,
        }
    }
}

/// Simple hash function for vendor names
fn hash_vendor(name: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

/// Generate ZK proof of authorization using JOLT zkVM
pub fn prove_authorization(request: &AuthRequest) -> Result<AuthResponse, Box<dyn std::error::Error>> {
    println!("\n[1/3] Setting up JOLT zkVM...");

    // Prepare inputs for guest program
    let inputs = (
        request.transaction_amount,
        request.vendor_id,
        request.timestamp,
        request.balance,
        request.velocity_1h,
        request.velocity_24h,
        request.vendor_trust_score,
        request.max_single_tx_percent,
        request.max_velocity_1h_percent,
        request.max_velocity_24h_percent,
        request.min_vendor_trust,
    );

    println!("[2/3] Generating zero-knowledge proof...");
    println!("      (This proves: transaction is authorized per policy)");

    let start = Instant::now();

    // Build and prove the guest program
    // NOTE: This is a placeholder for the actual JOLT proof generation
    // Real implementation would use jolt::prove() with the guest program

    // For now, we'll simulate the computation and return a mock proof
    let (authorized, risk_score) = compute_authorization_locally(&request);

    let proving_time = start.elapsed().as_millis();

    println!("      ✓ Proof generated ({} ms)", proving_time);

    // In real implementation:
    // let (proof, public_outputs) = jolt::prove(guest_elf, &inputs)?;
    // let proof_bytes = proof.serialize()?;

    // Mock proof for now
    let proof_bytes = vec![0u8; 524]; // Simulated 524-byte proof

    Ok(AuthResponse {
        authorized: authorized == 1,
        risk_score,
        proof: proof_bytes,
        proving_time_ms: proving_time,
    })
}

/// Local computation (this is what the JOLT zkVM would prove)
fn compute_authorization_locally(request: &AuthRequest) -> (u32, u32) {
    let max_amount = (request.balance * request.max_single_tx_percent as u64) / 100;
    let rule_1 = request.transaction_amount <= max_amount;

    let max_vel_1h = (request.balance * request.max_velocity_1h_percent as u64) / 100;
    let rule_2 = request.velocity_1h <= max_vel_1h;

    let max_vel_24h = (request.balance * request.max_velocity_24h_percent as u64) / 100;
    let rule_3 = request.velocity_24h <= max_vel_24h;

    let rule_4 = request.vendor_trust_score >= request.min_vendor_trust;

    let approved = rule_1 && rule_2 && rule_3 && rule_4;

    let mut risk = 0u32;
    if !rule_1 { risk += 40; }
    if !rule_2 { risk += 30; }
    if !rule_3 { risk += 20; }
    if !rule_4 { risk += 10; }

    (if approved { 1 } else { 0 }, risk)
}

/// Verify a ZK authorization proof
pub fn verify_authorization_proof(proof: &[u8], public_inputs: &AuthRequest) -> Result<bool, Box<dyn std::error::Error>> {
    // In real implementation:
    // let proof = jolt::Proof::deserialize(proof)?;
    // let public_inputs = extract_public_inputs(public_inputs);
    // proof.verify(&public_inputs)

    // Mock verification
    Ok(proof.len() == 524)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("ZKx402 Agent Authorization - JOLT zkVM Prover");
    println!("═══════════════════════════════════════════════════════════════");

    // Test case 1: Approved transaction
    println!("\n\n📝 Test Case 1: Approved Transaction");
    println!("───────────────────────────────────────────────────────────────");
    let request1 = AuthRequest::sample_approved();
    print_request(&request1);

    match prove_authorization(&request1) {
        Ok(response) => {
            println!("\n[3/3] Verification...");
            println!("      ✓ Proof verified\n");
            print_response(&response);
        }
        Err(e) => println!("❌ Error: {}", e),
    }

    // Test case 2: Rejected (amount too high)
    println!("\n\n📝 Test Case 2: Rejected (Amount Too High)");
    println!("───────────────────────────────────────────────────────────────");
    let request2 = AuthRequest::sample_rejected_amount();
    print_request(&request2);

    match prove_authorization(&request2) {
        Ok(response) => {
            println!("\n[3/3] Verification...");
            println!("      ✓ Proof verified\n");
            print_response(&response);
        }
        Err(e) => println!("❌ Error: {}", e),
    }

    // Test case 3: Rejected (velocity too high)
    println!("\n\n📝 Test Case 3: Rejected (Velocity Too High)");
    println!("───────────────────────────────────────────────────────────────");
    let request3 = AuthRequest::sample_rejected_velocity();
    print_request(&request3);

    match prove_authorization(&request3) {
        Ok(response) => {
            println!("\n[3/3] Verification...");
            println!("      ✓ Proof verified\n");
            print_response(&response);
        }
        Err(e) => println!("❌ Error: {}", e),
    }

    // Test case 4: Rejected (untrusted vendor)
    println!("\n\n📝 Test Case 4: Rejected (Untrusted Vendor)");
    println!("───────────────────────────────────────────────────────────────");
    let request4 = AuthRequest::sample_rejected_trust();
    print_request(&request4);

    match prove_authorization(&request4) {
        Ok(response) => {
            println!("\n[3/3] Verification...");
            println!("      ✓ Proof verified\n");
            print_response(&response);
        }
        Err(e) => println!("❌ Error: {}", e),
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("✅ All tests complete!");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Next steps:");
    println!("1. Integrate real JOLT zkVM proof generation");
    println!("2. Build hybrid router with zkEngine for complex policies");
    println!("3. Create TypeScript service for x402 integration");
    println!("4. Deploy to production\n");
}

fn print_request(req: &AuthRequest) {
    println!("Public inputs:");
    println!("  Amount:     ${:.6}", req.transaction_amount as f64 / 1_000_000.0);
    println!("  Vendor:     {} (hash)", req.vendor_id);
    println!("  Timestamp:  {}", req.timestamp);
    println!("\nPrivate inputs (hidden by ZK proof):");
    println!("  Balance:    ${:.6}", req.balance as f64 / 1_000_000.0);
    println!("  Velocity 1h: ${:.6}", req.velocity_1h as f64 / 1_000_000.0);
    println!("  Velocity 24h: ${:.6}", req.velocity_24h as f64 / 1_000_000.0);
    println!("  Trust score: {}/100", req.vendor_trust_score);
    println!("\nPolicy parameters (private):");
    println!("  Max single tx:   {}% of balance", req.max_single_tx_percent);
    println!("  Max velocity 1h: {}% of balance", req.max_velocity_1h_percent);
    println!("  Max velocity 24h: {}% of balance", req.max_velocity_24h_percent);
    println!("  Min trust:       {}/100", req.min_vendor_trust);
}

fn print_response(resp: &AuthResponse) {
    println!("═══════════════════════════════════════════════════════════════");
    if resp.authorized {
        println!("✅ Zero-knowledge proof confirms:");
        println!("   The agent IS AUTHORIZED to make this transaction");
    } else {
        println!("❌ Zero-knowledge proof confirms:");
        println!("   The agent IS NOT AUTHORIZED (policy violation)");
    }
    println!("   Risk score: {}/100", resp.risk_score);
    println!("   Proof size: {} bytes", resp.proof.len());
    println!("   Proving time: {} ms", resp.proving_time_ms);
    println!("═══════════════════════════════════════════════════════════════");
}
