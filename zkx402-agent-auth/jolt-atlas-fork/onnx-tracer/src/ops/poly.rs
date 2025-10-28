use std::{
    any::Any,
    ops::{Add, Mul, Neg, Sub},
};

use crate::{
    //   circuit::layouts,
    tensor::{self, Tensor, TensorError},
    trace_types::ONNXOpcode,
};

use super::*;

/// An enum representing the operations that can be expressed as arithmetic (non
/// lookup) operations.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PolyOp<F: TensorType + PartialOrd> {
    MultiBroadcastTo {
        shape: Vec<usize>,
    },
    Einsum {
        equation: String,
    },
    Conv {
        kernel: Tensor<F>,
        bias: Option<Tensor<F>>,
        padding: [(usize, usize); 2],
        stride: (usize, usize),
    },
    Downsample {
        axis: usize,
        stride: usize,
        modulo: usize,
    },
    DeConv {
        kernel: Tensor<F>,
        bias: Option<Tensor<F>>,
        padding: [(usize, usize); 2],
        output_padding: (usize, usize),
        stride: (usize, usize),
    },
    Add,
    Sub,
    Neg,
    Mult,
    Div,
    Identity,
    Reshape(Vec<usize>),
    MoveAxis {
        source: usize,
        destination: usize,
    },
    Flatten(Vec<usize>),
    Pad([(usize, usize); 2]),
    Sum {
        axes: Vec<usize>,
    },
    MeanOfSquares {
        axes: Vec<usize>,
    },
    Prod {
        axes: Vec<usize>,
        len_prod: usize,
    },
    Pow(u32),
    Pack(u32, u32),
    GlobalSumPool,
    Concat {
        axis: usize,
    },
    Slice {
        axis: usize,
        start: usize,
        end: usize,
    },
    Iff,
    Resize {
        scale_factor: Vec<usize>,
    },
    Not,
    And,
    Or,
    Xor,
}

impl<F: TensorType + PartialOrd> From<&PolyOp<F>> for ONNXOpcode {
    fn from(value: &PolyOp<F>) -> Self {
        match value {
            PolyOp::Add => ONNXOpcode::Add,
            PolyOp::Sub => ONNXOpcode::Sub,
            PolyOp::Mult => ONNXOpcode::Mul,
            PolyOp::Div => ONNXOpcode::Div,
            PolyOp::Pow(_) => ONNXOpcode::Pow,
            PolyOp::Einsum { .. } => ONNXOpcode::MatMult,
            PolyOp::Sum { .. } => ONNXOpcode::Sum,
            PolyOp::MeanOfSquares { .. } => ONNXOpcode::MeanOfSquares,
            PolyOp::Reshape(..) => ONNXOpcode::Reshape,
            PolyOp::Iff => ONNXOpcode::Select,
            PolyOp::MultiBroadcastTo { .. } => ONNXOpcode::Broadcast,
            PolyOp::Slice { .. } => ONNXOpcode::Slice,
            PolyOp::Identity => ONNXOpcode::Identity,
            _ => {
                panic!("PolyOp {value:?} cannot be converted to ONNXOpcode",);
            }
        }
    }
}

impl<
        F: TensorType
            + PartialOrd
            + Send
            + Sync
            + From<i32>
            + Add<Output = F>
            + Mul<Output = F>
            + Sub<Output = F>
            + Neg<Output = F>
            + std::iter::Sum,
    > Op<F> for PolyOp<F>
where
    i32: std::convert::From<F>,
{
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        match &self {
            PolyOp::MultiBroadcastTo { shape } => format!("MULTIBROADCASTTO (shape={shape:?})"),
            PolyOp::MoveAxis { .. } => "MOVEAXIS".into(),
            PolyOp::Downsample { .. } => "DOWNSAMPLE".into(),
            PolyOp::Resize { .. } => "RESIZE".into(),
            PolyOp::Iff => "IFF".into(),
            PolyOp::Einsum { equation, .. } => format!("EINSUM {equation}"),
            PolyOp::Identity => "IDENTITY".into(),
            PolyOp::Reshape(shape) => format!("RESHAPE (shape={shape:?})"),
            PolyOp::Flatten(_) => "FLATTEN".into(),
            PolyOp::Pad(_) => "PAD".into(),
            PolyOp::Add => "ADD".into(),
            PolyOp::Mult => "MULT".into(),
            PolyOp::Sub => "SUB".into(),
            PolyOp::Div => "DIV".into(),
            PolyOp::Sum { .. } => "SUM".into(),
            PolyOp::MeanOfSquares { axes } => {
                format!("MEANOFSQUARES (axes={axes:?})")
            }
            PolyOp::Prod { .. } => "PROD".into(),
            PolyOp::Pow(_) => "POW".into(),
            PolyOp::Pack(_, _) => "PACK".into(),
            PolyOp::GlobalSumPool => "GLOBALSUMPOOL".into(),
            PolyOp::Conv { .. } => "CONV".into(),
            PolyOp::DeConv { .. } => "DECONV".into(),
            PolyOp::Concat { axis } => format!("CONCAT (axis={axis})"),
            PolyOp::Slice { axis, start, end } => {
                format!("SLICE (axis={axis}, start={start}, end={end})")
            }
            PolyOp::Neg => "NEG".into(),
            PolyOp::Not => "NOT".into(),
            PolyOp::And => "AND".into(),
            PolyOp::Or => "OR".into(),
            PolyOp::Xor => "XOR".into(),
        }
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let mut inputs = inputs.to_vec();
        let res = match &self {
            PolyOp::MultiBroadcastTo { shape } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch(
                        "multibroadcastto inputs".to_string(),
                    ));
                }
                inputs[0].expand(shape)
            }
            PolyOp::And => tensor::ops::and(&inputs[0], &inputs[1]),
            PolyOp::Or => tensor::ops::or(&inputs[0], &inputs[1]),
            PolyOp::Xor => tensor::ops::xor(&inputs[0], &inputs[1]),
            PolyOp::Not => tensor::ops::not(&inputs[0]),
            PolyOp::Downsample {
                axis,
                stride,
                modulo,
            } => tensor::ops::downsample(&inputs[0], *axis, *stride, *modulo),
            PolyOp::Resize { scale_factor } => tensor::ops::resize(&inputs[0], scale_factor),
            PolyOp::Iff => tensor::ops::iff(&inputs[0], &inputs[1], &inputs[2]),
            PolyOp::Einsum { equation } => {
                // Check if this is the MatMul pattern "mk,nk->mn" and apply power-of-two padding
                if equation == "mk,nk->mn" && cfg!(feature = "matmul_power_of_two_padding") {
                    einsum_matmul_mk_nk_mn_padded(equation, &inputs)
                } else {
                    tensor::ops::einsum(equation, &inputs)
                }
            }
            PolyOp::Identity => Ok(inputs[0].clone()),
            PolyOp::Reshape(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims)?;
                Ok(t)
            }
            PolyOp::MoveAxis {
                source,
                destination,
            } => inputs[0].move_axis(*source, *destination),
            PolyOp::Flatten(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims)?;
                Ok(t)
            }
            PolyOp::Pad(p) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pad inputs".to_string()));
                }
                tensor::ops::pad(&inputs[0], *p)
            }
            PolyOp::Add => tensor::ops::add(&inputs),
            PolyOp::Neg => tensor::ops::neg(&inputs[0]),
            PolyOp::Sub => tensor::ops::sub(&inputs),
            PolyOp::Mult => tensor::ops::mult(&inputs),
            PolyOp::Div => tensor::ops::div(&inputs),
            PolyOp::Conv {
                kernel: a,
                bias,
                padding,
                stride,
            } => {
                inputs.push(a.clone());
                if let Some(b) = bias {
                    inputs.push(b.clone());
                }
                tensor::ops::conv(&inputs, *padding, *stride)
            }
            PolyOp::DeConv {
                kernel: a,
                bias,
                padding,
                output_padding,
                stride,
            } => {
                inputs.push(a.clone());
                if let Some(b) = bias {
                    inputs.push(b.clone());
                }
                tensor::ops::deconv(&inputs, *padding, *output_padding, *stride)
            }
            PolyOp::Pack(base, scale) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pack inputs".to_string()));
                }

                tensor::ops::pack(&inputs[0], F::from(*base as i32), *scale)
            }
            PolyOp::Pow(u) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pow inputs".to_string()));
                }
                inputs[0].pow(*u)
            }
            PolyOp::Sum { axes } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("sum inputs".to_string()));
                }
                tensor::ops::sum_axes(&inputs[0], axes)
            }
            PolyOp::MeanOfSquares { axes } => {
                let x = inputs[0].clone();
                let x = x.map(|x| i32::from(x));
                Ok(tensor::ops::nonlinearities::mean_of_squares_axes(&x, axes).map(|x| F::from(x)))
            }
            PolyOp::Prod { axes, .. } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("prod inputs".to_string()));
                }
                tensor::ops::prod_axes(&inputs[0], axes)
            }
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::Concat { axis } => {
                tensor::ops::concat(&inputs.iter().collect::<Vec<_>>(), *axis)
            }
            PolyOp::Slice { axis, start, end } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("slice inputs".to_string()));
                }
                Ok(tensor::ops::slice(&inputs[0], axis, start, end)?)
            }
        }?;

        Ok(ForwardResult {
            output: res,
            intermediate_lookups: vec![],
        })
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let scale = match self {
            PolyOp::MultiBroadcastTo { .. } => in_scales[0],
            PolyOp::Xor | PolyOp::Or | PolyOp::And | PolyOp::Not => 0,
            PolyOp::Neg => in_scales[0],
            PolyOp::MoveAxis { .. } => in_scales[0],
            PolyOp::Downsample { .. } => in_scales[0],
            PolyOp::Resize { .. } => in_scales[0],
            PolyOp::Iff => in_scales[1],
            PolyOp::Einsum { .. } => {
                let mut scale = in_scales[0];
                for s in in_scales.iter().skip(1) {
                    scale += *s;
                }
                scale
            }
            PolyOp::Prod { len_prod, .. } => in_scales[0] * (*len_prod as crate::Scale),
            PolyOp::Sum { .. } => in_scales[0],
            PolyOp::MeanOfSquares { .. } => 2 * in_scales[0],
            PolyOp::Conv { kernel, bias, .. } => {
                let kernel_scale = match kernel.scale() {
                    Some(s) => s,
                    None => return Err("scale must be set for conv kernel".into()),
                };
                let output_scale = in_scales[0] + kernel_scale;
                if let Some(b) = bias {
                    let bias_scale = match b.scale() {
                        Some(s) => s,
                        None => return Err("scale must be set for conv bias".into()),
                    };
                    assert_eq!(output_scale, bias_scale);
                }
                output_scale
            }
            PolyOp::DeConv { kernel, bias, .. } => {
                let kernel_scale = match kernel.scale() {
                    Some(s) => s,
                    None => return Err("scale must be set for deconv kernel".into()),
                };
                let output_scale = in_scales[0] + kernel_scale;
                if let Some(b) = bias {
                    let bias_scale = match b.scale() {
                        Some(s) => s,
                        None => return Err("scale must be set for deconv bias".into()),
                    };
                    assert_eq!(output_scale, bias_scale);
                }
                output_scale
            }
            PolyOp::Add => {
                let mut scale_a = 0;
                let scale_b = in_scales[0];
                scale_a += in_scales[1];
                assert_eq!(scale_a, scale_b);
                scale_a
            }
            PolyOp::Sub => in_scales[0],
            PolyOp::Div => {
                let mut scale = in_scales[0];
                scale -= in_scales[1];
                scale
            }
            PolyOp::Mult => {
                let mut scale = in_scales[0];
                scale += in_scales[1];
                scale
            }
            PolyOp::Identity => in_scales[0],
            PolyOp::Reshape(_) | PolyOp::Flatten(_) => in_scales[0],
            PolyOp::Pad(_) => in_scales[0],
            PolyOp::Pow(pow) => in_scales[0] * (*pow as crate::Scale),
            PolyOp::Pack(_, _) => in_scales[0],
            PolyOp::GlobalSumPool => in_scales[0],
            PolyOp::Concat { axis: _ } => in_scales[0],
            PolyOp::Slice { .. } => in_scales[0],
        };
        Ok(scale)
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        if matches!(self, PolyOp::Add | PolyOp::Sub | PolyOp::Concat { .. }) {
            vec![0, 1]
        } else if matches!(self, PolyOp::Iff) {
            vec![1, 2]
        } else {
            vec![]
        }
    }

    fn requires_shape_equality(&self) -> bool {
        matches!(
            self,
            PolyOp::Identity
                | PolyOp::Add
                | PolyOp::Neg
                | PolyOp::Sub
                | PolyOp::Mult
                | PolyOp::And
                | PolyOp::Or
                | PolyOp::Xor
                | PolyOp::Not
                | PolyOp::Iff
                | PolyOp::Pow(_)
        )
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// MatMul-specific power-of-two padding for the "mk,nk->mn" einsum pattern.
///
/// This function implements power-of-two padding specifically for matrix multiplication
/// operations of the form "mk,nk->mn", where:
/// - Input A: [m, k]
/// - Input B: [n, k]
/// - Output: [m, n]
///
/// The function pads M, N, K dimensions to the next power of two, performs the einsum
/// operation on the padded tensors, and then crops the result back to the original
/// expected output dimensions.
///
/// # Arguments
/// * `equation` - The einsum equation (should be "mk,nk->mn")
/// * `inputs` - Input tensors [A, B] where A is [m,k] and B is [n,k]
///
/// # Returns
/// * `Ok(Tensor<T>)` - Result tensor with dimensions [m, n]
/// * `Err(TensorError)` - If padding, computation, or cropping fails
fn einsum_matmul_mk_nk_mn_padded<T>(
    equation: &str,
    inputs: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError>
where
    T: TensorType + Mul<Output = T> + Add<Output = T> + Send + Sync,
{
    if inputs.len() != 2 {
        return Err(TensorError::DimMismatch(
            "mk,nk->mn requires exactly 2 input tensors".to_string(),
        ));
    }

    let a = &inputs[0]; // [m, k]
    let b = &inputs[1]; // [n, k]

    if a.dims().len() != 2 || b.dims().len() != 2 {
        return Err(TensorError::DimMismatch(
            "mk,nk->mn requires 2D input tensors".to_string(),
        ));
    }

    let m = a.dims()[0];
    let k_a = a.dims()[1];
    let n = b.dims()[0];
    let k_b = b.dims()[1];

    if k_a != k_b {
        return Err(TensorError::DimMismatch(
            "k dimensions must match for mk,nk->mn".to_string(),
        ));
    }

    let k = k_a;

    // Calculate power-of-two dimensions
    let m_pow2 = if m.is_power_of_two() {
        m
    } else {
        m.next_power_of_two()
    };
    let n_pow2 = if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    };
    let k_pow2 = if k.is_power_of_two() {
        k
    } else {
        k.next_power_of_two()
    };

    // Clone and pad input tensors
    let mut a_padded = a.clone();
    let mut b_padded = b.clone();

    a_padded.pad_to_dims(&[m_pow2, k_pow2])?;
    b_padded.pad_to_dims(&[n_pow2, k_pow2])?;

    // Perform einsum on padded tensors
    let padded_result = tensor::ops::einsum(equation, &[a_padded, b_padded])?;
    // Crop result back to original expected dimensions [m, n]
    padded_result.crop_to_dims(&[m, n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_einsum_matmul_mk_nk_mn_padded() {
        // Test the power-of-two padding function for MatMul
        // Input A: [3, 2] (m=3, k=2)
        // Input B: [2, 2] (n=2, k=2)
        // Expected output: [3, 2] (m=3, n=2)

        let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
        let b = Tensor::<i32>::new(Some(&[7, 8, 9, 10]), &[2, 2]).unwrap();

        let result = einsum_matmul_mk_nk_mn_padded("mk,nk->mn", &[a.clone(), b.clone()]).unwrap();

        // Verify output dimensions
        assert_eq!(result.dims(), &[3, 2]);

        // Compare with regular einsum result (should be identical)
        let regular_result = tensor::ops::einsum("mk,nk->mn", &[a, b]).unwrap();
        assert_eq!(result, regular_result);
    }

    #[test]
    fn test_einsum_matmul_different_sizes() {
        // Test with different matrix sizes to verify power-of-two padding works

        // Case 1: Already power-of-two (should be no-op)
        let a1 = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
        let b1 = Tensor::<i32>::new(Some(&[5, 6, 7, 8]), &[2, 2]).unwrap();
        let result1 =
            einsum_matmul_mk_nk_mn_padded("mk,nk->mn", &[a1.clone(), b1.clone()]).unwrap();
        let regular1 = tensor::ops::einsum("mk,nk->mn", &[a1, b1]).unwrap();
        assert_eq!(result1, regular1);

        // Case 2: Non-power-of-two dimensions
        let a2 =
            Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[3, 4]).unwrap(); // 3x4
        let b2 =
            Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[3, 4]).unwrap(); // 3x4
        let result2 =
            einsum_matmul_mk_nk_mn_padded("mk,nk->mn", &[a2.clone(), b2.clone()]).unwrap();
        let regular2 = tensor::ops::einsum("mk,nk->mn", &[a2, b2]).unwrap();
        assert_eq!(result2, regular2);
        assert_eq!(result2.dims(), &[3, 3]); // m=3, n=3
    }

    #[test]
    fn test_einsum_matmul_error_cases() {
        // Test error handling

        // Wrong number of inputs
        let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
        assert!(einsum_matmul_mk_nk_mn_padded("mk,nk->mn", &[a.clone()]).is_err());

        // Mismatched k dimensions
        let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap(); // k=2
        let b = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap(); // k=3
        assert!(einsum_matmul_mk_nk_mn_padded("mk,nk->mn", &[a, b]).is_err());
    }
}
