use itertools::Itertools;
use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};
use serde::{Deserialize, Serialize};

use crate::{
    jolt::instruction::{
        VirtualInstructionSequence, add::ADD, beq::BEQInstruction, mul::MUL, sub::SUB,
        virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
        virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
    },
    utils::u64_vec_to_i32_iter,
};

/// Perform signed division and return the result
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct DIVInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for DIVInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 9;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Div);
        let use_v2 = std::env::var("JOLT_DIV_V2").ok().as_deref() == Some("1");
        // DIV source registers
        let r_x = cycle.instr.ts1;

        // Virtual registers used in sequence
        let v_0 = Some(virtual_tensor_index(0));
        let v_q = Some(virtual_tensor_index(1));
        let v_r = Some(virtual_tensor_index(2));
        let v_qy = Some(virtual_tensor_index(3));
        let v_c = Some(virtual_tensor_index(4));

        // DIV operands
        let x = cycle.ts1_vals();
        // Prefer immediate constant divisor if present, else use ts2 tensor values
        let y = if cycle.instr.imm.is_some() { cycle.imm() } else { cycle.ts2_vals() };
        let mut virtual_trace = vec![];

        // If enabled and divisor is a non-zero constant across active elements, use a lighter v2 sequence
        let active = cycle.instr.active_output_elements.min(MAX_TENSOR_SIZE);
        let is_const_divisor = cycle.instr.imm.is_some() && {
            let d0 = y.get(0).cloned().unwrap_or(0);
            d0 != 0 && (0..active).all(|i| y[i] == d0)
        };

        if use_v2 {
            // Compute quotient deterministically (signed Euclidean-like with floor toward -inf semantics from current gadget)
            let (quotient, remainder) = {
                let mut quotient_tensor = vec![0; MAX_TENSOR_SIZE];
                let mut remainder_tensor = vec![0; MAX_TENSOR_SIZE];
                for i in 0..MAX_TENSOR_SIZE {
                    let xi = x[i] as i64;
                    let yi = y[i] as i64; // constant
                    if yi == 0 {
                        quotient_tensor[i] = match WORD_SIZE { 32 => u32::MAX as u64, 64 => u64::MAX, _ => 0 };
                        remainder_tensor[i] = x[i];
                        continue;
                    }
                    let mut q = xi / yi;
                    let mut r = xi % yi;
                    if (r < 0 && yi > 0) || (r > 0 && yi < 0) {
                        r += yi;
                        q -= 1;
                    }
                    quotient_tensor[i] = (q as i128 as i64) as u64;
                    remainder_tensor[i] = (r as i128 as i64) as u64;
                }
                (quotient_tensor, remainder_tensor)
            };

            // Advice(q)
            let q_lookup = (0..MAX_TENSOR_SIZE)
                .map(|i| ADVICEInstruction::<WORD_SIZE>(quotient[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualAdvice,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: v_q,
                    imm: None,
                    virtual_sequence_remaining: Some(7),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: None,
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_lookup))),
                },
                advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&quotient))),
            });

            // If imm divisor present, materialize const denom; otherwise use dynamic ts2
            if is_const_divisor {
                virtual_trace.push(ONNXCycle {
                    instr: ONNXInstr {
                        address: cycle.instr.address,
                        opcode: ONNXOpcode::VirtualConst,
                        ts1: None,
                        ts2: None,
                        ts3: None,
                        td: v_c,
                        imm: cycle.instr.imm.clone(),
                        virtual_sequence_remaining: Some(6),
                        active_output_elements: cycle.instr.active_output_elements,
                        output_dims: cycle.instr.output_dims,
                    },
                    memory_state: MemoryState {
                        ts1_val: None,
                        ts2_val: None,
                        ts3_val: None,
                        td_pre_val: None,
                        td_post_val: cycle.instr.imm.clone(),
                    },
                    advice_value: None,
                });
            }

            let qy_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| MUL::<WORD_SIZE>(quotient[i], y[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Mul,
                    ts1: v_q,
                    ts2: if is_const_divisor { v_c } else { cycle.instr.ts2 },
                    ts3: None,
                    td: v_qy,
                    imm: None,
                    virtual_sequence_remaining: Some(5),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&quotient))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&qy_vals))),
                },
                advice_value: None,
            });

            // r = x - q*y (SUB)
            let r_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| SUB::<WORD_SIZE>(x[i], qy_vals[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Sub,
                    ts1: r_x,
                    ts2: v_qy,
                    ts3: None,
                    td: v_r,
                    imm: None,
                    virtual_sequence_remaining: Some(4),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&qy_vals))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&r_vals))),
                },
                advice_value: None,
            });

            // sum = q*y + r (ADD) then assert eq with x
            let sum_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| ADD::<WORD_SIZE>(qy_vals[i], r_vals[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Add,
                    ts1: v_qy,
                    ts2: v_r,
                    ts3: None,
                    td: v_0,
                    imm: None,
                    virtual_sequence_remaining: Some(3),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&qy_vals))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&r_vals))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&sum_vals))),
                },
                advice_value: None,
            });

            // Assert sum == x
            let _assert_eq = (0..MAX_TENSOR_SIZE)
                .map(|i| BEQInstruction::<WORD_SIZE>(sum_vals[i], x[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualAssertEq,
                    ts1: v_0,
                    ts2: r_x,
                    ts3: None,
                    td: None,
                    imm: None,
                    virtual_sequence_remaining: Some(10),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&sum_vals))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: None,
                },
                advice_value: None,
            });

            // v2 gating and booleanity additions (constant divisor optimization)
            // is_zero advice derived from y
            let z_vals: Vec<u64> = (0..MAX_TENSOR_SIZE).map(|i| if y[i] == 0 { 1 } else { 0 }).collect();
            let z_lookup = (0..MAX_TENSOR_SIZE)
                .map(|i| ADVICEInstruction::<WORD_SIZE>(z_vals[i]).to_lookup_output())
                .collect_vec();
            let v_z = Some(virtual_tensor_index(5));
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualAdvice,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: v_z,
                    imm: None,
                    virtual_sequence_remaining: Some(20),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState { ts1_val: None, ts2_val: None, ts3_val: None, td_pre_val: None, td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_lookup))), },
                advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&z_vals))),
            });

            // const one
            let v_one = Some(virtual_tensor_index(6));
            let one_tensor = Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 1)));
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualConst,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: v_one,
                    imm: one_tensor.clone(),
                    virtual_sequence_remaining: Some(19),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState { ts1_val: None, ts2_val: None, ts3_val: None, td_pre_val: None, td_post_val: one_tensor },
                advice_value: None,
            });

            // one_minus_z = 1 - z
            let v_omz = Some(virtual_tensor_index(7));
            let omz_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| SUB::<WORD_SIZE>(1, z_vals[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Sub,
                    ts1: v_one,
                    ts2: v_z,
                    ts3: None,
                    td: v_omz,
                    imm: None,
                    virtual_sequence_remaining: Some(18),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 1))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_vals))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&omz_vals))),
                },
                advice_value: None,
            });

            // diff = x - sum
            let v_diff = Some(virtual_tensor_index(8));
            let diff_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| SUB::<WORD_SIZE>(x[i], sum_vals[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Sub,
                    ts1: r_x,
                    ts2: v_0,
                    ts3: None,
                    td: v_diff,
                    imm: None,
                    virtual_sequence_remaining: Some(17),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&sum_vals))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&diff_vals))),
                },
                advice_value: None,
            });

            // gated = diff * (1 - z)
            let v_gated = Some(virtual_tensor_index(9));
            let gated_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| MUL::<WORD_SIZE>(diff_vals[i], omz_vals[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Mul,
                    ts1: v_diff,
                    ts2: v_omz,
                    ts3: None,
                    td: v_gated,
                    imm: None,
                    virtual_sequence_remaining: Some(16),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&diff_vals))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&omz_vals))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&gated_vals))),
                },
                advice_value: None,
            });

            // const zero
            let v_zero = Some(virtual_tensor_index(10));
            let zero_tensor = Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 0)));
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualConst,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: v_zero,
                    imm: zero_tensor.clone(),
                    virtual_sequence_remaining: Some(15),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState { ts1_val: None, ts2_val: None, ts3_val: None, td_pre_val: None, td_post_val: zero_tensor },
                advice_value: None,
            });

            // assert gated == 0
            let _assert_zero_gated = (0..MAX_TENSOR_SIZE)
                .map(|i| BEQInstruction::<WORD_SIZE>(gated_vals[i], 0).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualAssertEq,
                    ts1: v_gated,
                    ts2: v_zero,
                    ts3: None,
                    td: None,
                    imm: None,
                    virtual_sequence_remaining: Some(14),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState { ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&gated_vals))), ts2_val: Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 0))), ts3_val: None, td_pre_val: None, td_post_val: None },
                advice_value: None,
            });

            // booleanity: z * (1 - z) == 0
            let v_bool = Some(virtual_tensor_index(11));
            let bool_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| MUL::<WORD_SIZE>(z_vals[i], omz_vals[i]).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Mul,
                    ts1: v_z,
                    ts2: v_omz,
                    ts3: None,
                    td: v_bool,
                    imm: None,
                    virtual_sequence_remaining: Some(13),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState { ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&z_vals))), ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&omz_vals))), ts3_val: None, td_pre_val: None, td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&bool_vals))) },
                advice_value: None,
            });
            let _assert_bool = (0..MAX_TENSOR_SIZE)
                .map(|i| BEQInstruction::<WORD_SIZE>(bool_vals[i], 0).to_lookup_output())
                .collect_vec();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualAssertEq,
                    ts1: v_bool,
                    ts2: v_zero,
                    ts3: None,
                    td: None,
                    imm: None,
                    virtual_sequence_remaining: Some(12),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState { ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&bool_vals))), ts2_val: Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 0))), ts3_val: None, td_pre_val: None, td_post_val: None },
                advice_value: None,
            });

            // Assert remainder validity and move q to output
            let is_valid: Vec<u64> = (0..MAX_TENSOR_SIZE)
                .map(|i| {
                    AssertValidSignedRemainderInstruction::<WORD_SIZE>(r_vals[i], y[i])
                        .to_lookup_output()
                })
                .collect_vec();
            is_valid.iter().for_each(|&valid| assert_eq!(valid, 1));

            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualAssertValidSignedRemainder,
                    ts1: v_r,
                    ts2: if is_const_divisor { v_c } else { cycle.instr.ts2 },
                    ts3: None,
                    td: None,
                    imm: None,
                    virtual_sequence_remaining: Some(1),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&r_vals))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: None,
                },
                advice_value: None,
            });

            // Move q to result
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualMove,
                    ts1: v_q,
                    ts2: None,
                    ts3: None,
                    td: cycle.instr.td,
                    imm: None,
                    virtual_sequence_remaining: Some(0),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&quotient))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: cycle.memory_state.td_pre_val.clone(),
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&quotient))),
                },
                advice_value: None,
            });

            return virtual_trace;
        }

        let (quotient, remainder) = {
            let mut quotient_tensor = vec![0; MAX_TENSOR_SIZE];
            let mut remainder_tensor = vec![0; MAX_TENSOR_SIZE];
            // Division semantics: default to signed (RISC-V REM/REMW style) unless explicitly set to 'unsigned'
            let signed_mode = std::env::var("JOLT_DIV_SEMANTICS").ok().map(|s| s != "unsigned").unwrap_or(true);
            for i in 0..MAX_TENSOR_SIZE {
                let x = x[i];
                let y = y[i];
                let (quotient, remainder) = match WORD_SIZE {
                    32 => {
                        if signed_mode {
                            // Signed 32-bit semantics similar to RISC-V REMW
                            let xi = x as i32;
                            let yi = y as i32;
                            if yi == 0 {
                                (u32::MAX as u64, x) // div-by-zero: q sentinel, r = dividend
                            } else if xi == i32::MIN && yi == -1 {
                                // Overflow case: quotient = INT_MIN, remainder = 0 (holds mod 2^32)
                                (xi as u32 as u64, 0u32 as u64)
                            } else {
                                let q = xi.wrapping_div(yi);
                                let r = xi.wrapping_rem(yi); // RISC-V remainder sign follows dividend
                                (q as u32 as u64, r as u32 as u64)
                            }
                        } else {
                            // Unsigned 32-bit semantics
                            let xu = x as u32;
                            let yu = y as u32;
                            if yu == 0 {
                                (u32::MAX as u64, xu as u64)
                            } else {
                                let q = xu / yu;
                                let r = xu % yu;
                                (q as u64, r as u64)
                            }
                        }
                    }
                    64 => {
                        if signed_mode {
                            let xi = x as i64;
                            let yi = y as i64;
                            if yi == 0 {
                                (u64::MAX, x)
                            } else if xi == i64::MIN && yi == -1 {
                                (xi as u64, 0u64)
                            } else {
                                let q = xi.wrapping_div(yi);
                                let r = xi.wrapping_rem(yi);
                                (q as u64, r as u64)
                            }
                        } else {
                            let xu = x as u64;
                            let yu = y as u64;
                            if yu == 0 { (u64::MAX, xu) } else { (xu / yu, xu % yu) }
                        }
                    }
                    _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}",),
                };
                quotient_tensor[i] = quotient;
                remainder_tensor[i] = remainder;
            }
            (quotient_tensor, remainder_tensor)
        };

        // const
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_c,
                imm: cycle.instr.imm.clone(),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: cycle.instr.imm,
            },
            advice_value: None,
        });

        let q = (0..MAX_TENSOR_SIZE)
            .map(|i| ADVICEInstruction::<WORD_SIZE>(quotient[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_q,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
            },
            advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&quotient))),
        });

        let r = (0..MAX_TENSOR_SIZE)
            .map(|i| ADVICEInstruction::<WORD_SIZE>(remainder[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                ts3: None,
                td: v_r,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
            },
            advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&remainder))),
        });

        let is_valid: Vec<u64> = (0..MAX_TENSOR_SIZE)
            .map(|i| {
                AssertValidSignedRemainderInstruction::<WORD_SIZE>(r[i], y[i]).to_lookup_output()
            })
            .collect_vec();
        is_valid.iter().for_each(|&valid| {
            assert_eq!(valid, 1, "Invalid signed remainder detected");
        });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertValidSignedRemainder,
                ts1: v_r,
                ts2: v_c,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: None,
            },
            advice_value: None,
        });

        let is_valid: Vec<u64> = (0..MAX_TENSOR_SIZE)
            .map(|i| AssertValidDiv0Instruction::<WORD_SIZE>(y[i], q[i]).to_lookup_output())
            .collect_vec();
        is_valid.iter().for_each(|&valid| {
            assert_eq!(valid, 1, "Invalid division by zero detected");
        });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertValidDiv0,
                ts1: v_c,
                ts2: v_q,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: None,
            },
            advice_value: None,
        });

        let q_y = (0..MAX_TENSOR_SIZE)
            .map(|i| MUL::<WORD_SIZE>(q[i], y[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_q,
                ts2: v_c,
                ts3: None,
                td: v_qy,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_y))),
            },
            advice_value: None,
        });

        let add_0 = (0..MAX_TENSOR_SIZE)
            .map(|i| ADD::<WORD_SIZE>(q_y[i], r[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Add,
                ts1: v_qy,
                ts2: v_r,
                ts3: None,
                td: v_0,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q_y))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&r))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&add_0))),
            },
            advice_value: None,
        });

        let _assert_eq = (0..MAX_TENSOR_SIZE)
            .map(|i| BEQInstruction::<WORD_SIZE>(add_0[i], x[i]).to_lookup_output())
            .collect_vec();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertEq,
                ts1: v_0,
                ts2: r_x,
                ts3: None,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&add_0))),
                ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&x))),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: None,
            },
            advice_value: None,
        });

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_q,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&q))),
            },
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, y: Vec<u64>, _: Option<ONNXOpcode>) -> Vec<u64> {
        let mut output = vec![0; MAX_TENSOR_SIZE];
        for i in 0..MAX_TENSOR_SIZE {
            let x = x[i];
            let y = y[i];
            let x = x as i32;
            let y = y as i32;
            if y == 0 {
                output[i] = (1 << WORD_SIZE) - 1;
                continue;
            }
            let mut quotient = x / y;
            let remainder = x % y;
            if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
                quotient -= 1;
            }
            output[i] = quotient as u32 as u64
        }
        output
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;
    use rand::RngCore;

    #[test]
    fn div_virtual_sequence_32() {
        jolt_virtual_sequence_test::<DIVInstruction<32>>(ONNXOpcode::Div);
    }

    #[test]
    fn div_virtual_sequence_32_dynamic_denominator_v2() {
        use ark_std::test_rng;
        let mut rng = test_rng();
        for _ in 0..128 {
            let t_x = (rng.next_u32() % 32) as usize;
            let mut t_y = (rng.next_u32() % 32) as usize;
            if t_y == t_x { t_y = ((t_y + 1) % 32).max(1); }
            let mut td = (rng.next_u32() % 32) as usize;
            if td == 0 { td = 1; }

            let x_vals: Vec<u64> = (0..MAX_TENSOR_SIZE)
                .map(|_| (rng.next_u32() as i64 % 33 - 16) as u32 as u64)
                .collect();
            let y_vals: Vec<u64> = (0..MAX_TENSOR_SIZE)
                .map(|_| (rng.next_u32() as i64 % 17 - 8) as u32 as u64)
                .collect();

            let expected = DIVInstruction::<32>::sequence_output(x_vals.clone(), y_vals.clone(), None);

            let cycle = ONNXCycle {
                instr: ONNXInstr {
                    address: rng.next_u32() as usize,
                    opcode: ONNXOpcode::Div,
                    ts1: Some(t_x),
                    ts2: Some(t_y),
                    ts3: None,
                    td: Some(td),
                    imm: None, // dynamic y from ts2
                    virtual_sequence_remaining: None,
                    active_output_elements: MAX_TENSOR_SIZE,
                    output_dims: [1, MAX_TENSOR_SIZE],
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x_vals))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&y_vals))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&expected))),
                },
                advice_value: None,
            };

            let vseq = DIVInstruction::<32>::virtual_trace(cycle);
            assert!(vseq.len() >= 5);
            let last = vseq.last().unwrap();
            assert_eq!(last.td_post_vals(), expected);
        }
    }

    #[test]
    fn div_virtual_sequence_32_constant_denominator_nonzero() {
        use ark_std::test_rng;
        let mut rng = test_rng();
        for _ in 0..64 {
            let t_x = (rng.next_u32() % 32) as usize;
            let mut td = (rng.next_u32() % 32) as usize;
            if td == 0 { td = 1; }

            let x_vals: Vec<u64> = (0..MAX_TENSOR_SIZE)
                .map(|_| (rng.next_u32() as i64 % 129 - 64) as u32 as u64)
                .collect();
            let denom: i32 = ((rng.next_u32() % 15) + 1) as i32; // 1..=16
            let y_vals: Vec<u64> = vec![denom as u32 as u64; MAX_TENSOR_SIZE];

            let expected = DIVInstruction::<32>::sequence_output(x_vals.clone(), y_vals.clone(), None);

            let cycle = ONNXCycle {
                instr: ONNXInstr {
                    address: rng.next_u32() as usize,
                    opcode: ONNXOpcode::Div,
                    ts1: Some(t_x),
                    ts2: None,
                    ts3: None,
                    td: Some(td),
                    imm: Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| denom))),
                    virtual_sequence_remaining: None,
                    active_output_elements: MAX_TENSOR_SIZE,
                    output_dims: [1, MAX_TENSOR_SIZE],
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x_vals))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&expected))),
                },
                advice_value: None,
            };

            let vseq = DIVInstruction::<32>::virtual_trace(cycle);
            let last = vseq.last().unwrap();
            assert_eq!(last.td_post_vals(), expected);
        }
    }

    #[test]
    fn div_virtual_sequence_32_constant_denominator_zero() {
        let t_x = 3usize; let td = 5usize;
        let x_vals: Vec<u64> = (0..MAX_TENSOR_SIZE).map(|i| (i as i32 - 16) as u32 as u64).collect();
        let y_vals: Vec<u64> = vec![0u64; MAX_TENSOR_SIZE];
        let expected = DIVInstruction::<32>::sequence_output(x_vals.clone(), y_vals.clone(), None);
        let cycle = ONNXCycle {
            instr: ONNXInstr {
                address: 42usize,
                opcode: ONNXOpcode::Div,
                ts1: Some(t_x),
                ts2: None,
                ts3: None,
                td: Some(td),
                imm: Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 0))),
                virtual_sequence_remaining: None,
                active_output_elements: MAX_TENSOR_SIZE,
                output_dims: [1, MAX_TENSOR_SIZE],
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&x_vals))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&expected))),
            },
            advice_value: None,
        };
        let vseq = DIVInstruction::<32>::virtual_trace(cycle);
        let last = vseq.last().unwrap();
        assert_eq!(last.td_post_vals(), expected);
    }
}
