# Archived Documentation

This directory contains documentation from previous iterations of the zkX402 project.

## Project Evolution

### Phase 1: ZK-Fair-Pricing (Archived)
- Used zkEngine WASM for pricing proofs
- x402 payment protocol integration
- Documented in: TECHNICAL_SPEC.md, QUICKSTART.md, X402_*.md

### Phase 2: Dual System (Archived)
- Both fair-pricing and agent authorization
- Multiple proof systems (zkEngine + JOLT Atlas)
- Documented in: USE_CASES_OVERVIEW.md, DEMO_COMBINED.md

### Current: JOLT Atlas Agent Authorization Only
- Focus exclusively on agent authorization using enhanced JOLT Atlas
- See main README.md for current documentation

## Archived Files

- **FINAL_SUMMARY.md** - Original project summary including fair-pricing
- **QUICKSTART.md** - Old quickstart with zkEngine setup
- **TECHNICAL_SPEC.md** - zkEngine WASM technical specification
- **DEMO_COMBINED.md** - Combined demo of both systems
- **X402_BAZAAR_INTEGRATION.md** - Marketplace integration (fair-pricing focused)
- **X402_DEEP_INTEGRATION.md** - x402 protocol deep dive
- **USE_CASES_OVERVIEW.md** - Comparison of both use cases
- **ZK_AGENT_AUTHORIZATION.md** - Agent auth business case (pre-refactor)
- **GITHUB_SETUP.md** - GitHub setup with both systems
- **BUILD_COMPLETE.md** - Build status for old architecture
- **SUMMARY.md** - Old project summary

## Why Archived?

The project has been refocused to specialize in **zero-knowledge machine learning authorization** using JOLT Atlas. The fair-pricing use case, while functional, was moved to archived status to streamline the project focus.

## Legacy Code Removed

- `zkEngine_dev/` - WASM-based proof system (521MB)
- `zkx402-service/` - TypeScript x402 service layer

These have been removed from the repository but remain in git history if needed.

---

**Date Archived**: 2025-10-28
**Current Focus**: JOLT Atlas zkML Agent Authorization Only
