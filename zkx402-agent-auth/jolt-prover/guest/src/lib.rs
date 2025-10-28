#![cfg_attr(feature = "guest", no_std)]
#![no_main]

/// Guest program for velocity-based spending authorization
///
/// This program runs inside the JOLT zkVM and proves that a transaction
/// is authorized according to a velocity policy WITHOUT revealing:
/// - The user's balance
/// - Recent spending history
/// - Velocity limits
///
/// Public inputs:
/// - transaction_amount (micro-USDC)
/// - vendor_id (hash)
/// - timestamp
///
/// Private inputs:
/// - balance
/// - velocity_1h
/// - velocity_24h
/// - vendor_trust_score
/// - policy_params (thresholds)
///
/// Output:
/// - authorized: bool (0 or 1)
/// - risk_score: u32 (0-100)

#[jolt::provable]
fn check_authorization(
    // Public inputs
    transaction_amount: u64,
    vendor_id: u64,
    timestamp: u64,
    // Private inputs (hidden by ZK proof)
    balance: u64,
    velocity_1h: u64,
    velocity_24h: u64,
    vendor_trust_score: u32, // 0-100
    // Policy parameters (private)
    max_single_tx_percent: u32,   // e.g., 10 = 10% of balance
    max_velocity_1h_percent: u32, // e.g., 5 = 5% of balance per hour
    max_velocity_24h_percent: u32, // e.g., 20 = 20% of balance per day
    min_vendor_trust: u32,        // e.g., 50 = minimum trust score of 50/100
) -> (u32, u32) {
    // Rule 1: Transaction amount < max_single_tx_percent of balance
    let max_amount = (balance * max_single_tx_percent as u64) / 100;
    let rule_1 = transaction_amount <= max_amount;

    // Rule 2: 1h velocity < max_velocity_1h_percent of balance
    let max_vel_1h = (balance * max_velocity_1h_percent as u64) / 100;
    let rule_2 = velocity_1h <= max_vel_1h;

    // Rule 3: 24h velocity < max_velocity_24h_percent of balance
    let max_vel_24h = (balance * max_velocity_24h_percent as u64) / 100;
    let rule_3 = velocity_24h <= max_vel_24h;

    // Rule 4: Vendor trust score > minimum
    let rule_4 = vendor_trust_score >= min_vendor_trust;

    // Approved if ALL rules pass
    let approved = rule_1 && rule_2 && rule_3 && rule_4;

    // Calculate risk score (0-100)
    let mut risk = 0u32;
    if !rule_1 {
        risk += 40; // High risk: excessive amount
    }
    if !rule_2 {
        risk += 30; // Medium risk: high short-term velocity
    }
    if !rule_3 {
        risk += 20; // Low risk: high daily velocity
    }
    if !rule_4 {
        risk += 10; // Low risk: untrusted vendor
    }

    // Return (authorized, risk_score)
    // authorized: 0 = rejected, 1 = approved
    // risk_score: 0 = safe, 100 = maximum risk
    (if approved { 1 } else { 0 }, risk)
}
