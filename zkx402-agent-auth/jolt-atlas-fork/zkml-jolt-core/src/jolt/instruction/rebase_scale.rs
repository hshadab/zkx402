use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};
use jolt_core::jolt::instruction::LookupQuery;

use crate::{
    jolt::{
        instruction::{VirtualInstructionSequence, div::DIVInstruction},
        precompiles::matmult::{MatMultPrecompile, MatMultPrecompileDims},
    },
    utils::u64_vec_to_i32_iter,
};

macro_rules! expect_rebase_scale {
    ($cycle:expr) => {
        match $cycle.instr.opcode {
            ONNXOpcode::RebaseScale(_) => {}
            _ => panic!("Expected ONNXOpcode::RebaseScale"),
        }
    };
}

/// Perform signed division and return the result
pub struct REBASEInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for REBASEInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 1 + DIVInstruction::<WORD_SIZE>::SEQUENCE_LENGTH;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        expect_rebase_scale!(cycle);

        let inner_opcode = match &cycle.instr.opcode {
            ONNXOpcode::RebaseScale(inner) => inner,
            _ => unreachable!(),
        };

        // Virtual registers used in sequence
        let v_0 = Some(virtual_tensor_index(0)); // Inner operator output to be rebased 

        let mut virtual_trace = vec![];

        // Apply inner operator
        let inner_opcode = (**inner_opcode).clone();
        let mut instr = cycle.instr.clone();
        instr.opcode = inner_opcode;
        let inner_res_0: Vec<u64> = {
            let x = cycle.ts1_vals();
            let y = cycle.ts2_vals();
            match instr.opcode {
                ONNXOpcode::MatMult => {
                    // For MatMult: A[m,k] × B[k,n] = C[m,n]
                    // We need k for the precompile dims.

                    // Debug: Print tensor shapes
                    eprintln!("\n=== MatMult Debug ===");
                    if let Some(ts1) = &cycle.memory_state.ts1_val {
                        eprintln!("ts1 (input) dims: {:?}", ts1.dims());
                        eprintln!("ts1 first 10 values: {:?}", &x[..10.min(x.len())]);
                    }
                    if let Some(ts2) = &cycle.memory_state.ts2_val {
                        eprintln!("ts2 (weight) dims: {:?}", ts2.dims());
                        eprintln!("ts2 first 10 values: {:?}", &y[..10.min(y.len())]);
                    }
                    eprintln!("output_dims from instr: {:?}", instr.output_dims);

                    // Handle both 2D [m, n] and 1D [n] output shapes
                    let (m, n) = if instr.output_dims.len() >= 2 {
                        (instr.output_dims[0], instr.output_dims[1])
                    } else if instr.output_dims.len() == 1 {
                        // 1D output: treat as [1, n] (single row)
                        (1, instr.output_dims[0])
                    } else {
                        panic!("Invalid output_dims length: {:?}", instr.output_dims);
                    };

                    // Try ts2 first (the weight matrix), then ts1, handling 1D vectors
                    let k = cycle
                        .memory_state
                        .ts2_val
                        .as_ref()
                        .and_then(|tensor| {
                            let dims = tensor.dims();
                            if dims.len() > 1 {
                                Some(dims[0])  // For transposed weight [k,n], first dim is k
                            } else if dims.len() == 1 {
                                Some(dims[0])  // 1D vector
                            } else {
                                None
                            }
                        })
                        .or_else(|| {
                            cycle.memory_state.ts1_val.as_ref().and_then(|tensor| {
                                let dims = tensor.dims();
                                if dims.len() > 1 {
                                    Some(dims[1])  // For input [m,k], second dim is k
                                } else {
                                    None
                                }
                            })
                        })
                        .unwrap_or(1);

                    let dims: MatMultPrecompileDims = (m, n, k);

                    eprintln!("Computed dims (m, n, k): {:?}", dims);

                    let (result, _shape) =
                        MatMultPrecompile::new(x, y).matmult_rhs_transposed(dims);

                    eprintln!("MatMult result first 10: {:?}", &result[..10.min(result.len())]);

                    let mut output = result
                        .iter()
                        .map(|&x| x as u32 as u64)
                        .collect::<Vec<u64>>();
                    output.resize(MAX_TENSOR_SIZE, 0);

                    eprintln!("Output after resize first 10: {:?}", &output[..10]);
                    eprintln!("===================\n");

                    output
                }
                ONNXOpcode::Mul => match WORD_SIZE {
                    8 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u8).wrapping_mul(b as u8) as u64)
                        .collect::<Vec<u64>>(),
                    32 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u32).wrapping_mul(b as u32) as u64)
                        .collect::<Vec<u64>>(),
                    64 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| a.wrapping_mul(b))
                        .collect::<Vec<u64>>(),
                    _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
                },
                _ => panic!("Unimplemented inner opcode: {:?}", instr.opcode),
            }
        };

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: instr.opcode,
                ts1: cycle.instr.ts1,
                ts2: cycle.instr.ts2,
                ts3: cycle.instr.ts3,
                td: v_0,
                imm: cycle.instr.imm.clone(),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: cycle.memory_state.ts1_val.clone(),
                ts2_val: cycle.memory_state.ts2_val.clone(),
                ts3_val: cycle.memory_state.ts3_val.clone(),
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&inner_res_0))), // Needs to set to the inner operator's expected output
            },
            advice_value: None,
        });
        // Apply div operator by 2^scale
        let res = DIVInstruction::<WORD_SIZE>::sequence_output(
            inner_res_0.clone(),
            vec![128; MAX_TENSOR_SIZE],
            None,
        );

        // Debug: Compare computed vs expected
        let expected = cycle.td_post_vals();
        eprintln!("\n=== Assertion Check ===");
        eprintln!("inner_res_0 (before div) first 10: {:?}", &inner_res_0[..10]);
        eprintln!("res (after div by 128) first 10: {:?}", &res[..10]);
        eprintln!("expected (cycle.td_post_vals) first 10: {:?}", &expected[..10]);
        if res != expected {
            eprintln!("❌ MISMATCH DETECTED!");
            eprintln!("Diff in first 20 elements:");
            for i in 0..20.min(res.len()) {
                if res[i] != expected[i] {
                    eprintln!("  [{}] computed={} expected={}", i, res[i], expected[i]);
                }
            }
        } else {
            eprintln!("✅ Match!");
        }
        eprintln!("===================\n");

        assert_eq!(res, expected);

        let div_cycle = ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_0,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![
                    128;
                    MAX_TENSOR_SIZE
                ]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
                output_dims: cycle.instr.output_dims,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&inner_res_0))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
            },
            advice_value: None,
        };

        let rewrite = std::env::var("JOLT_REWRITE_CONST_DIV").ok().as_deref() == Some("1");
        if rewrite {
            use crate::jolt::instruction::{virtual_advice::ADVICEInstruction, mul::MUL, beq::BEQInstruction};
            use onnx_tracer::constants::virtual_tensor_index;

            // v_r = advised res
            let v_r = Some(virtual_tensor_index(1));
            let r_lookup = (0..MAX_TENSOR_SIZE)
                .map(|i| ADVICEInstruction::<WORD_SIZE>(res[i]).to_lookup_output())
                .collect::<Vec<u64>>();
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
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&r_lookup))),
                },
                advice_value: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
            });

            // v_c = const 128
            let v_c = Some(virtual_tensor_index(2));
            let denom_tensor = Some(Tensor::from(u64_vec_to_i32_iter(&vec![128; MAX_TENSOR_SIZE])));
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualConst,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: v_c,
                    imm: denom_tensor.clone(),
                    virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState { ts1_val: None, ts2_val: None, ts3_val: None, td_pre_val: None, td_post_val: denom_tensor },
                advice_value: None,
            });

            // v_prod = v_r * v_c
            let v_prod = Some(virtual_tensor_index(3));
            let prod_vals = (0..MAX_TENSOR_SIZE)
                .map(|i| MUL::<WORD_SIZE>(res[i], 128).to_lookup_output())
                .collect::<Vec<u64>>();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Mul,
                    ts1: v_r,
                    ts2: v_c,
                    ts3: None,
                    td: v_prod,
                    imm: None,
                    virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&vec![128; MAX_TENSOR_SIZE]))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&prod_vals))),
                },
                advice_value: None,
            });

            // assert v_prod == inner_res_0
            let _assert_ok = (0..MAX_TENSOR_SIZE)
                .map(|i| BEQInstruction::<WORD_SIZE>(prod_vals[i], inner_res_0[i]).to_lookup_output())
                .collect::<Vec<u64>>();
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualAssertEq,
                    ts1: v_prod,
                    ts2: v_0,
                    ts3: None,
                    td: None,
                    imm: None,
                    virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&prod_vals))),
                    ts2_val: Some(Tensor::from(u64_vec_to_i32_iter(&inner_res_0))),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: None,
                },
                advice_value: None,
            });

            // Move advised result to output
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualMove,
                    ts1: v_r,
                    ts2: None,
                    ts3: None,
                    td: cycle.instr.td,
                    imm: None,
                    virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                    active_output_elements: cycle.instr.active_output_elements,
                    output_dims: cycle.instr.output_dims,
                },
                memory_state: MemoryState {
                    ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
                },
                advice_value: None,
            });
        } else {
            let div_virtual_trace = DIVInstruction::<WORD_SIZE>::virtual_trace(div_cycle);
            virtual_trace.extend(div_virtual_trace);
        }

        virtual_trace
    }

    fn sequence_output(x: Vec<u64>, y: Vec<u64>, inner: Option<ONNXOpcode>) -> Vec<u64> {
        if inner.is_none() {
            panic!("Inner opcode must be provided for RebaseScale");
        }
        let Some(inner_opcode) = inner else {
            unreachable!()
        };

        match inner_opcode {
            ONNXOpcode::Mul => {
                let mul_res = match WORD_SIZE {
                    8 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u8).wrapping_mul(b as u8) as u64)
                        .collect::<Vec<u64>>(),
                    32 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| (a as u32).wrapping_mul(b as u32) as u64)
                        .collect::<Vec<u64>>(),
                    64 => x
                        .iter()
                        .zip(y.iter())
                        .map(|(&a, &b)| a.wrapping_mul(b))
                        .collect::<Vec<u64>>(),
                    _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
                };
                DIVInstruction::<WORD_SIZE>::sequence_output(
                    mul_res,
                    vec![128; MAX_TENSOR_SIZE],
                    None,
                )
            }
            // TODO: Need to check other possible inner opcodes
            _ => panic!("Unimplemented inner opcode: {inner_opcode:?}"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    #[test]
    fn rebasescale_virtual_sequence_32() {
        jolt_virtual_sequence_test::<REBASEInstruction<32>>(ONNXOpcode::RebaseScale(Box::new(
            ONNXOpcode::Mul,
        )));
    }
}
