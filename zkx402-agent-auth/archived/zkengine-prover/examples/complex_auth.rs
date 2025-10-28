use zk_engine::{
    args::WASMArgsBuilder,
    nova::{
        provider::{ipa_pc, Bn256EngineIPA},
        spartan,
        traits::Dual,
    },
    wasm::{WasmSNARK, args::WASMCtx},
    traits::snark::RelaxedR1CSSNARKTrait,
};

/// Type aliases for zkEngine (same as zkx402_pricing example)
pub type E = Bn256EngineIPA;
pub type EE1 = ipa_pc::EvaluationEngine<E>;
pub type EE2 = ipa_pc::EvaluationEngine<Dual<E>>;
pub type S1 = spartan::batched::BatchedRelaxedR1CSSNARK<E, EE1>;
pub type S2 = spartan::batched::BatchedRelaxedR1CSSNARK<Dual<E>, EE2>;

/// Authorization request for complex policy
#[derive(Debug)]
pub struct ComplexAuthRequest {
    // Public inputs
    pub transaction_amount: u64, // micro-USDC
    pub vendor_id: u64,          // hash of vendor domain
    pub timestamp: u64,          // Unix timestamp

    // Private inputs (hidden by ZK proof)
    pub balance: u64,                 // micro-USDC
    pub daily_budget_remaining: u64,  // micro-USDC
    pub whitelist_bitmap: u64,        // 64-bit bitmap of allowed vendors
}

impl ComplexAuthRequest {
    /// Approved: Small amount, whitelisted vendor, business hours, budget available
    pub fn sample_approved() -> Self {
        Self {
            transaction_amount: 500_000,   // $0.50
            vendor_id: 5,                   // Vendor ID 5 (will be in whitelist)
            timestamp: 1704117600,          // Wed Jan 1 2025, 2pm EST (business hours)
            balance: 10_000_000,            // $10.00
            daily_budget_remaining: 2_000_000, // $2.00 remaining
            whitelist_bitmap: 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00100000, // Bit 5 set
        }
    }

    /// Rejected: Vendor not whitelisted
    pub fn sample_rejected_vendor() -> Self {
        Self {
            transaction_amount: 500_000,
            vendor_id: 10,                  // Vendor ID 10 (NOT in whitelist)
            timestamp: 1704117600,
            balance: 10_000_000,
            daily_budget_remaining: 2_000_000,
            whitelist_bitmap: 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00100000, // Only bit 5 set
        }
    }

    /// Rejected: Outside business hours (weekend)
    pub fn sample_rejected_time() -> Self {
        Self {
            transaction_amount: 500_000,
            vendor_id: 5,
            timestamp: 1704585600,          // Sat Jan 6 2025 (weekend)
            balance: 10_000_000,
            daily_budget_remaining: 2_000_000,
            whitelist_bitmap: 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00100000,
        }
    }

    /// Rejected: Exceeds daily budget
    pub fn sample_rejected_budget() -> Self {
        Self {
            transaction_amount: 500_000,
            vendor_id: 5,
            timestamp: 1704117600,
            balance: 10_000_000,
            daily_budget_remaining: 100_000, // Only $0.10 remaining (need $0.50)
            whitelist_bitmap: 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00100000,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("ZKx402 Agent Authorization - zkEngine Complex Policy Prover");
    println!("═══════════════════════════════════════════════════════════════");

    let step_size = 100; // Adjust based on circuit complexity

    // Test case 1: Approved
    println!("\n\n📝 Test Case 1: Approved Transaction");
    println!("───────────────────────────────────────────────────────────────");
    let request1 = ComplexAuthRequest::sample_approved();
    run_proof(&request1, step_size)?;

    // Test case 2: Rejected (vendor)
    println!("\n\n📝 Test Case 2: Rejected (Vendor Not Whitelisted)");
    println!("───────────────────────────────────────────────────────────────");
    let request2 = ComplexAuthRequest::sample_rejected_vendor();
    run_proof(&request2, step_size)?;

    // Test case 3: Rejected (time)
    println!("\n\n📝 Test Case 3: Rejected (Outside Business Hours)");
    println!("───────────────────────────────────────────────────────────────");
    let request3 = ComplexAuthRequest::sample_rejected_time();
    run_proof(&request3, step_size)?;

    // Test case 4: Rejected (budget)
    println!("\n\n📝 Test Case 4: Rejected (Exceeds Daily Budget)");
    println!("───────────────────────────────────────────────────────────────");
    let request4 = ComplexAuthRequest::sample_rejected_budget();
    run_proof(&request4, step_size)?;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("✅ All tests complete!");
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

fn run_proof(request: &ComplexAuthRequest, step_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    print_request(request);

    println!("\n[1/4] Loading WASM circuit...");
    let wasm_path = "wasm/authorization.wat";

    // Build WASM context with function and inputs
    let wasm_ctx = WASMCtx::new_from_file(
        WASMArgsBuilder::default()
            .file_path(wasm_path)
            .invoke("check_authorization")  // Call the main authorization function
            .param(request.transaction_amount)
            .param(request.vendor_id)
            .param(request.timestamp)
            .param(request.balance)
            .param(request.daily_budget_remaining)
            .param(request.whitelist_bitmap)
            .build(),
    )?;

    // Set memory step size for efficient proving
    wasm_ctx.set_memory_step_size(50_000);

    println!("      ✓ WASM circuit loaded");
    println!("      ✓ Function: check_authorization");
    println!("      ✓ Inputs: 6 parameters");

    println!("\n[2/4] Generating public parameters (step_size={})...", step_size);
    println!("      (This is a one-time setup per circuit)");

    let pp = WasmSNARK::<E, S1, S2>::setup(&wasm_ctx, step_size)?;

    println!("      ✓ Public parameters generated");

    println!("\n[3/4] Generating zero-knowledge proof...");
    println!("      (This proves: transaction authorization is valid per policy)");

    let (snark, instance) = WasmSNARK::<E, S1, S2>::prove(&pp, &wasm_ctx, step_size)?;

    println!("      ✓ Proof generated");

    // Extract result from instance
    let authorized = if !instance.is_empty() {
        // The output is the last element of the instance
        let result_idx = instance.len() - 1;
        instance[result_idx] != E::Scalar::ZERO
    } else {
        false
    };

    println!("\n[4/4] Verifying proof...");

    snark.verify(&pp, &instance)?;

    println!("      ✓ Proof verified");

    print_result(authorized);

    Ok(())
}

fn print_request(req: &ComplexAuthRequest) {
    println!("Public inputs:");
    println!("  Amount:     ${:.6}", req.transaction_amount as f64 / 1_000_000.0);
    println!("  Vendor ID:  {}", req.vendor_id);
    println!("  Timestamp:  {} ({})", req.timestamp, format_timestamp(req.timestamp));
    println!("\nPrivate inputs (hidden by ZK proof):");
    println!("  Balance:           ${:.6}", req.balance as f64 / 1_000_000.0);
    println!("  Daily budget left: ${:.6}", req.daily_budget_remaining as f64 / 1_000_000.0);
    println!("  Whitelist:         0b{:064b}", req.whitelist_bitmap);
    println!("\nPolicy rules (enforced in circuit):");
    println!("  1. Amount <= 10% of balance");
    println!("  2. Vendor must be whitelisted");
    println!("  3. Transaction during business hours (Mon-Fri, 9am-5pm EST)");
    println!("  4. Amount <= daily budget remaining");
}

fn print_result(authorized: bool) {
    println!("\n═══════════════════════════════════════════════════════════════");
    if authorized {
        println!("✅ Zero-knowledge proof confirms:");
        println!("   The agent IS AUTHORIZED to make this transaction");
        println!("   All policy rules passed:");
        println!("   ✓ Amount within limit");
        println!("   ✓ Vendor whitelisted");
        println!("   ✓ Business hours");
        println!("   ✓ Budget available");
    } else {
        println!("❌ Zero-knowledge proof confirms:");
        println!("   The agent IS NOT AUTHORIZED (policy violation)");
        println!("   At least one policy rule failed");
    }
    println!("═══════════════════════════════════════════════════════════════");
}

fn format_timestamp(timestamp: u64) -> String {
    let days_since_epoch = timestamp / 86400;
    let day_of_week = (days_since_epoch + 4) % 7; // Unix epoch was Thursday
    let hour = (timestamp % 86400) / 3600;

    let day_name = match day_of_week {
        0 => "Sun",
        1 => "Mon",
        2 => "Tue",
        3 => "Wed",
        4 => "Thu",
        5 => "Fri",
        6 => "Sat",
        _ => "?",
    };

    format!("{} {:02}:00", day_name, hour)
}
