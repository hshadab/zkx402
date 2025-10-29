//! # Node Module for ONNX Computational Graphs
//!
//! This module defines the core data structures and logic for representing and manipulating nodes within a computational graph,
//! specifically tailored for ONNX model tracing and quantized execution in the zkML-Jolt framework.
//!
//! ## Purpose
//!
//! The `node` module is essential for modeling the computation graph of a neural network or other ONNX-based models.
//!  Each node encapsulates an operation (such as a layer or mathematical function),
//! its input/output relationships, quantization scale, and metadata required for fixed-point arithmetic and zero-knowledge proof compatibility.
//!
//! ## Overview of Components
//!
//! - **Node Structure:** Represents a single operation in the computation graph, including its operation type (`SupportedOp`), input/output connections, quantization scale, output dimensions, and usage count.
//! - **SupportedOp Enum:** Enumerates all supported operation types, including linear, nonlinear, hybrid, input, constant, and special wrappers for rescaling and rebasing scales.
//! - **Rescaled & RebaseScale Wrappers:** Provide mechanisms for adjusting the scale of operations and outputs to ensure consistent fixed-point precision across the graph.
//! - **Node Construction Logic:** Handles parsing ONNX nodes, propagating and homogenizing input scales, rescaling constants, and determining output shapes and scales.
//! - **ONNX Instruction Decoding:** Allows nodes to be converted into ONNX instructions for downstream processing or execution in a zkVM context.
//!
//! ## Usage
//!
//! This module is typically used as part of the ONNX model import and tracing pipeline. When an ONNX model is loaded, each ONNX node is converted into a `Node` instance using the `Node::new` constructor. The resulting graph of `Node`s is then used for quantized inference, circuit synthesis, or zero-knowledge proof generation.
//!
//! Example usage context:
//! 1. **Model Import:** Parse an ONNX model and construct a computation graph of `Node` instances.
//! 2. **Quantization & Scale Propagation:** Ensure all nodes operate on compatible fixed-point scales, automatically rescaling constants and outputs as needed.
//! 3. **Graph Traversal & Execution:** Traverse the graph to perform inference, generate zkVM instructions, or synthesize circuits for proof generation.
//!
//! ## Context
//!
//! This module is a foundational part of the `onnx-tracer` crate within the zkML-Jolt project. It interacts closely with:
//! - The `model` and `vars` modules for graph-wide metadata and variable scale management.
//! - The `ops` module for operation implementations.
//! - The `tensor` module for tensor arithmetic and quantization utilities.
//! - The `trace_types` module for ONNX instruction and opcode representations.
//!
//! By abstracting the details of node construction, scale management, and operation decoding, this module enables robust and efficient handling of ONNX models in privacy-preserving and quantized computation settings.

use crate::{
    constants::MAX_TENSOR_SIZE,
    graph::{
        model::NodeType,
        utilities::{
            multiplier_to_scale, new_op_from_onnx, node_output_shapes, quantize_tensor,
            scale_to_multiplier,
        },
        vars::VarScales,
    },
    ops::{
        hybrid::HybridOp, lookup::LookupOp, poly::PolyOp, Constant, ForwardResult, Input, Op,
        Unknown,
    },
    tensor::{Tensor, TensorError},
    trace_types::{ONNXInstr, ONNXOpcode},
};
use log::{trace, warn};
use std::{collections::BTreeMap, error::Error, fmt, fmt::Debug};
use tabled::Tabled;
use tract_onnx::{
    self,
    prelude::{
        tract_itertools::Itertools, Node as OnnxNode, OutletId, SymbolValues, TypedFact, TypedOp,
    },
};

/// Represents a node output connection as (node_index, output_slot).
/// A node's input is a tensor from another node's output.
pub type Outlet = (usize, usize);

#[derive(Clone, Debug, Default)]
/// A single operation in a [crate::graph::Model].
/// Represents a node in the computation graph, encapsulating an operation and its associated metadata.
///
/// # Fields
///
/// - `opkind`: The operation this node performs, represented by the [`SupportedOp`] enum.
/// - `out_scale`: The denominator for the fixed-point representation of the node's output. This is used for quantization purposes; nodes with different output scales should not be combined directly.
/// - `inputs`: A list of [`Outlet`]s, each representing a connection from another node's output to this node's input.
///   - **Purpose:** The `inputs` field defines the data dependencies for this node. Each entry specifies which node and which output of that node is used as an input.
///   - **When to use:** Use `inputs` when constructing or traversing the computation graph to determine the source of input data for this node.
///   - **How to use:** Each `Outlet` in the vector identifies a specific output from another node (by node index and output slot). To fetch the input tensor for this node, follow the corresponding `Outlet` to the producing node's output.
///   - **What is `Outlet`:** An `Outlet` is a reference to a specific output of another node in the graph, typically containing the producing node's index and the output slot index. This allows for flexible graph topologies, including nodes with multiple outputs or inputs.
/// - `out_dims`: The shape (dimensions) of the output tensor produced by this node.
/// - `idx`: The unique identifier for this node within the graph.
/// - `num_uses`: The number of times this node's output is consumed by other nodes (i.e., how many downstream nodes depend on this node).
pub struct Node {
    /// [Op] i.e what operation this node represents.
    pub opkind: SupportedOp,
    /// The fixed-point output scale (denominator) for this node's output tensor.
    ///
    /// This value represents the scaling factor (denominator) used in fixed-point quantization for the output tensor of this node.
    /// In quantized neural networks, real numbers are represented as integers scaled by a fixed denominator ("scale").
    /// For example, if `out_scale = 1000`, then the integer value `1234` represents the real value `1.234`.
    ///
    /// - Ensures all tensors in the computation graph use compatible fixed-point representations.
    /// - Prevents arithmetic errors when combining or comparing tensors from different nodes.
    /// - Allows precise control over quantization error and numeric precision.
    ///
    /// # How is it calculated?
    /// - For most nodes, the output scale is determined by the operation and the input scales, following quantization rules.
    /// - For constants and inputs, it is set during quantization or model import.
    /// - For nodes that require output scale alignment (e.g., for addition or concatenation), the scale may be "rebased" to a global maximum to ensure consistency.
    ///
    /// - Think of this as the "precision" of the node's output: higher values mean more decimal places are preserved.
    /// - All nodes that directly feed into each other should have matching or compatible `out_scale` values.
    /// - When debugging quantization issues, mismatches in `out_scale` are a common source of errors.
    ///
    /// # Important:
    /// - Never combine tensors with different `out_scale` values without rescaling.
    /// - This field is critical for correct, efficient, and numerically stable quantized inference.
    pub out_scale: i32,
    /// A list of [`Outlet`]s representing the sources of input tensors for this node.
    ///
    /// Each entry in this vector specifies a connection from another node's output to this node's input.
    /// An [`Outlet`] is a tuple of (node_index, output_slot), where:
    ///   - `node_index` refers to the index of the producing node in the computation graph.
    ///   - `output_slot` specifies which output of the producing node is being used (for nodes with multiple outputs).
    ///
    /// This field defines the data dependencies for this node: to compute its output, the node will
    /// fetch tensors from the outputs of the nodes referenced here. The order of the vector corresponds
    /// to the order in which the operation expects its inputs (e.g., for a binary operation, the first
    /// entry is the left operand, the second is the right operand).
    ///
    /// Example:
    ///   - For an affine (fully connected) node, `inputs` might contain three entries:
    ///     1. The output of the previous layer (input tensor)
    ///     2. The weights tensor
    ///     3. The bias tensor
    ///   - For a unary operation (like ReLU), `inputs` will typically have a single entry.
    ///
    /// This design allows for flexible and dynamic graph topologies, including support for nodes with
    /// multiple inputs and outputs, and enables traversal or analysis of the graph structure by following
    /// these connections.
    pub inputs: Vec<Outlet>,
    /// Dimensions of output.
    pub out_dims: Vec<usize>,
    /// The node's unique identifier.
    pub idx: usize,
    /// The node's num of uses
    pub num_uses: usize,
}

impl Node {
    /// Constructs a new [`Node`] from a given tract [`OnnxNode`], integrating it into the computational graph.
    ///
    /// # Arguments
    /// * `node` - The tract [`OnnxNode`] to convert.
    /// * `other_nodes` - A mutable reference to a [`BTreeMap`] containing previously initialized [`Node`]s in the graph.
    /// * `scales` - Reference to [`VarScales`] for managing scale propagation.
    /// * `idx` - The unique identifier for the new node.
    /// * `symbol_values` - Reference to [`SymbolValues`] for resolving symbolic dimensions.
    ///
    /// # Returns
    /// Returns a new [`Node`] instance representing the converted ONNX node, with its operation, inputs, output shape, scale, and usage count properly set up.
    ///
    /// # Step-by-step Functionality
    /// 0. **Input Collection:** Collects the indices of input nodes and retrieves their corresponding [`Node`] objects from `other_nodes`.
    /// 1. **Operation Construction:** Calls `new_op_from_onnx` to parse and construct the operation (`opkind`) for this node, and identifies any unused input indices.
    /// 2. **Node Map Update:** Updates `other_nodes` with any new or modified input nodes.
    /// 3. **Input Pruning:** Marks and removes unused inputs based on `deleted_indices`.
    /// 4. **Input Scale Gathering:** Collects the output scales of each input node to prepare for scale propagation.
    /// 5. **Constant Rescaling:** For operations requiring homogeneous input scales, automatically rescales constants that are only used once to match the required scale, updating the corresponding input nodes.
    /// 6. **Homogeneous Rescaling:** Applies homogeneous rescaling to the operation if required, ensuring all input scales are consistent.
    /// 7. **Global Scale Rebase:** Rebases the operation's output scale to the global maximum scale, ensuring consistent fixed-point precision across the graph.
    /// 8. **Output Shape Resolution:** Determines the output shape of the node using `node_output_shapes`, resolving symbolic dimensions as needed.
    /// 9. **Output Shape Adjustment:** Ensures the output shape is non-empty, defaulting to `[1]` if necessary.
    /// 10. **Node Construction:** Returns a new [`Node`] instance with all fields (index, operation, inputs, output dimensions, output scale, and usage count) properly initialized.
    ///
    /// # Panics
    /// Panics if any required input node is missing from `other_nodes`, or if operation construction or scale propagation fails.
    ///
    /// # Notes
    /// - This function not only constructs a new [`Node`], but also mutates `other_nodes` to reflect any changes to input nodes (e.g., after rescaling constants).
    /// - The function ensures that all scale and shape propagation is handled consistently, which is critical for correct and efficient graph execution.
    pub fn new(
        node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
        other_nodes: &mut BTreeMap<usize, NodeType>,
        scales: &VarScales,
        symbol_values: &SymbolValues,
        remappings: &BTreeMap<usize, usize>,
    ) -> Self {
        trace!("Create {node:?}",);
        trace!("Create op {:?}", node.op);
        let idx = other_nodes.len();
        // Determine how many times this node's output is used in the graph.
        // This is important for optimizations such as rescaling constants:
        // if a constant is only used once, we can safely rescale it in-place
        // to match the consumer's scale, which avoids unnecessary rescaling ops.
        // For output nodes (with no successors), we ensure num_uses is at least 1.
        let num_uses = std::cmp::max(
            node.outputs
                .iter()
                .map(|outlet| outlet.successors.len())
                .sum::<usize>(),
            // cmp to 1 for outputs
            1,
        );
        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 0: Gather input nodes and their identifiers for this ONNX node ★
        // ──────────────────────────────────────────────────────────────────────────────
        // WHY:
        //   - Every node in a computational graph performs an operation on one or more input tensors.
        //   - These input tensors are produced by other nodes in the graph (its "predecessors").
        //   - To construct this node and its operation, we need to know:
        //       1. Which nodes produce its inputs (by index and output slot).
        //       2. The actual Node objects for those inputs, so we can access their metadata
        //          (such as output scale, shape, etc.) for scale propagation, shape inference,
        //          and operation construction.
        //
        // WHAT:
        //   - We create two collections:
        //       a) `input_ids`: a vector of (node index, output slot) pairs for each input.
        //          This is a lightweight reference to the source of each input tensor.
        //       b) `inputs`: a vector of the actual Node objects corresponding to those inputs.
        //          This allows us to access all relevant information about each input node.
        //
        // HOW:
        //   1. We iterate over the ONNX node's `inputs` field, which lists its input connections.
        //      Each input is an object with a `.node` (index of the producing node) and `.slot`
        //      (which output of that node is used, for multi-output nodes).
        //   2. We collect these into `input_ids`, which will later be used to reference and update
        //      the connections for this node.
        //   3. For each input node index, we fetch the corresponding Node object from `other_nodes`
        //      (the global map of all nodes constructed so far) and push a clone of it into `inputs`.
        //      This gives us direct access to all input node metadata for downstream processing.
        //
        //   Note: We must collect the input node indices first, because we can only borrow `other_nodes`
        //   mutably once per function scope. By collecting the indices up front, we avoid borrowing issues
        //   and can safely fetch all input nodes before any mutations occur.
        //
        //   This setup is foundational for the rest of the node construction process, as it enables:
        //     - Operation parsing (which may depend on input node properties)
        //     - Scale and shape propagation (which require input node metadata)
        //     - Input pruning and rescaling optimizations (which may mutate input nodes)
        //
        //   In summary: This block establishes the data dependencies for the new node, and prepares
        //   all necessary input node information for the subsequent steps of operation construction,
        //   scale handling, and graph mutation.
        let mut inputs = vec![];
        // Collect (node index, slot index) pairs for each input of the current node.
        let mut input_ids = map_outlet_indices(&node.inputs, remappings);
        // For each input node index, fetch the corresponding Node from `other_nodes` and push it to `inputs`.
        input_ids.iter().for_each(|(i, _)| {
            inputs.push(other_nodes.get(i).unwrap().clone());
        });
        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 1: Parse the ONNX node into an operation and identify unused inputs ★
        // ──────────────────────────────────────────────────────────────────────────────
        // `new_op_from_onnx` constructs the operation (`opkind`) for this node based on the ONNX node.
        // It also returns a list of input indices (`deleted_indices`) that are not actually used by the operation.
        // This is important for pruning unused inputs (e.g., optional bias in some layers).
        let (mut opkind, deleted_indices) =
            new_op_from_onnx(idx, scales, node.clone(), &mut inputs, symbol_values).unwrap();

        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 2: Update the global node map with any modified input nodes ★
        // ──────────────────────────────────────────────────────────────────────────────
        // Some input nodes may have been mutated (e.g., rescaled constants).
        // We update `other_nodes` with the latest versions of all input nodes.
        other_nodes.extend(
            inputs
                .iter()
                .map(|i| (i.idx(), i.clone()))
                .collect::<BTreeMap<_, _>>(),
        );

        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 3: Mark unused inputs for removal ★
        // ──────────────────────────────────────────────────────────────────────────────
        // For each input, if its index is in `deleted_indices`, we set its node index to `usize::MAX`.
        // This is a sentinel value indicating that this input should be ignored/removed.
        // The actual pruning happens later via `retain`.
        input_ids.iter_mut().enumerate().for_each(|(i, (idx, _))| {
            if deleted_indices.contains(&i) {
                // ✗ This input is not used by the operation; mark for removal.
                *idx = usize::MAX;
            }
        });
        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 4: Prune unused inputs and gather input scales for scale propagation ★
        // ──────────────────────────────────────────────────────────────────────────────
        // Remove any inputs that were marked as unused (idx == usize::MAX) in the previous step.
        // This ensures that only the relevant inputs are retained for this node's operation.
        input_ids.retain(|(idx, _)| *idx != usize::MAX);

        // For each retained input, determine its output scale.
        // This is necessary for scale propagation and for ensuring that all inputs to the operation
        // are quantized to compatible fixed-point representations.
        //
        // - We map each (node_idx, outlet) pair to the corresponding input node in `inputs`.
        // - For each input, we fetch the output scale for the specific outlet.
        let mut in_scales: Vec<crate::Scale> = input_ids
            .iter()
            .map(|(idx, outlet)| {
                // Find the position of the input node in the `inputs` vector.
                let idx = inputs.iter().position(|x| *idx == x.idx()).unwrap();
                // Get the output scale for the specific outlet of this input node.
                inputs[idx].out_scales()[*outlet]
            })
            .collect::<Vec<_>>();

        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 5: Homogenize input scales for operations that require it ★
        // ──────────────────────────────────────────────────────────────────────────────
        // Some operations require all their inputs to have the same scale (homogenous input scales).
        // - We query the operation for which input indices require homogenous scaling.
        // - For each such input, if it is a constant and is only used once, we can safely rescale it
        //   in-place to match the required scale, avoiding unnecessary rescaling operations.
        // - This optimization is important for efficiency and for minimizing quantization error.
        let homogenous_inputs = opkind.requires_homogenous_input_scales();

        // For each input that requires homogenous scaling and is not deleted:
        for input in homogenous_inputs
            .into_iter()
            .filter(|i| !deleted_indices.contains(i))
        {
            // Ensure the input index is valid.
            if inputs.len() > input {
                // Fetch the input node from the global node map.
                let input_node = other_nodes
                    .get_mut(&inputs[input].idx())
                    .ok_or("input not found")
                    .unwrap();
                // Get a mutable reference to the input node's operation.
                let input_opkind = &mut input_node.opkind();
                // If the input is a constant, and is only used once, rescale it in-place.
                if let Some(constant) = input_opkind.get_mutable_constant() {
                    rescale_const_with_single_use(
                        constant,
                        in_scales.clone(),
                        input_node.num_uses(),
                    )
                    .unwrap();
                    // Replace the input node's operation with the newly rescaled constant.
                    input_node.replace_opkind(constant.clone_dyn().into());
                    // Update the input node's output scale to reflect the new scale.
                    let out_scale = input_opkind.out_scale(vec![]).unwrap();
                    input_node.bump_scale(out_scale);
                    // Update the in_scales vector for this input.
                    in_scales[input] = out_scale;
                }
            } else {
                // If the input index is invalid, log a warning and skip.
                warn!("input {input} not found for rescaling, skipping ...",);
            }
        }

        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 6: Apply homogenous rescaling to the operation if required ★
        // ──────────────────────────────────────────────────────────────────────────────
        // If the operation requires homogenous input scales, apply the necessary rescaling.
        // This ensures that all inputs to the operation are quantized to the same scale,
        // which is required for correct computation in fixed-point arithmetic.
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            opkind = opkind.homogenous_rescale(in_scales.clone()).unwrap().into();
        }
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            // WASM fallback: skip rescale operation
            // The out_scale method has a different return type incompatible with .into()
            // For WASM, we'll skip this rescaling step
        }

        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 7: Compute the output scale for this node ★
        // ──────────────────────────────────────────────────────────────────────────────
        // The output scale is determined by the operation, given the input scales.
        // This is used for subsequent scale propagation and for quantizing the output tensor.
        let mut out_scale = opkind.out_scale(in_scales.clone()).unwrap();
        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 8: Rebase the output scale to the global maximum scale ★
        // ──────────────────────────────────────────────────────────────────────────────
        // Why: To ensure consistent fixed-point precision across the entire computation graph,
        //      we want all node outputs to use a common "global" scale (the highest precision required).
        //      This avoids subtle bugs and numerical errors when combining outputs from different nodes,
        //      and simplifies downstream processing (e.g., when exporting or verifying the model).
        //
        // What: If this node's output scale is higher than the global scale (i.e., it has more precision),
        //       we "rebase" it down to the global scale by wrapping the operation in a `RebaseScale` op.
        //       This applies a scaling multiplier to the output, so that its fixed-point representation
        //       matches the global scale. The `scales.rebase_multiplier` allows for additional scaling
        //       flexibility (e.g., for multi-scale quantization).
        //
        // How: We call `RebaseScale::rebase`, which checks if rebasing is needed and, if so, wraps the
        //      operation accordingly. We then update `out_scale` to reflect the new (rebased) scale.
        let global_scale = scales.get_max();
        // TODO(AntoineF4C5): Might have to handle out_scale < global_scale if `Div` nodes are used (for now seem to be replaced by mul with divisor's inverse)
        opkind = RebaseScale::rebase(opkind, global_scale, out_scale, scales.rebase_multiplier);
        out_scale = opkind.out_scale(in_scales).unwrap();

        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 9: Determine the output shape for this node ★
        // ──────────────────────────────────────────────────────────────────────────────
        // Why: Each node must know the shape of its output tensor for correct graph execution,
        //      shape inference, and for allocating memory buffers.
        //
        // What: We use `node_output_shapes` to compute the output shape(s) of this ONNX node,
        //       resolving any symbolic dimensions using `symbol_values`.
        //
        // How: Most nodes (except subgraphs) have a single output, so we take the first shape.
        //      If the output shape is empty (e.g., a scalar), we default to `[1]` to ensure
        //      downstream code always has a valid shape vector.
        let out_dims = node_output_shapes(&node, symbol_values).unwrap();
        let mut out_dims = out_dims[0].clone();
        if out_dims.is_empty() {
            out_dims = vec![1];
        }

        // ──────────────────────────────────────────────────────────────────────────────
        // ★ Step 10: Construct and return the new Node instance ★
        // ──────────────────────────────────────────────────────────────────────────────
        // Why: All fields are now fully determined—operation, input connections, output shape,
        //      output scale, and usage count—so we can safely construct the Node.
        //
        // What: The returned Node is ready for insertion into the computation graph, with all
        //       metadata and invariants satisfied.
        // #Note: If node has `Input` opkind, we set its idx to 0.
        let idx = if matches!(opkind, SupportedOp::Input(_)) {
            0
        } else {
            other_nodes.len()
        };
        Node {
            idx,
            opkind,
            inputs: input_ids,
            out_dims,
            out_scale,
            num_uses,
        }
    }

    /// Compares a Node to a `BTreeMap` of nodes to determine, for each of the node inputs,
    /// if its shape matches the corresponding input node's output shape.
    /// If the shapes do not match, it updates the BTreeMap to insert a "Broadcast" node
    /// mapping the input node's output to the required shape. It also updates the current node's input indexing.
    ///
    /// # Arguments
    ///
    /// * `other_nodes` - A mutable reference to a BTreeMap containing the nodes in the graph.
    ///
    /// # Returns
    /// The number of broadcast nodes added to the graph.
    pub fn homogenize_input_shapes(&mut self, other_nodes: &mut BTreeMap<usize, NodeType>) {
        for (j, input_outlet) in self.inputs.clone().iter().enumerate() {
            let input = if let NodeType::Node(n) = other_nodes.get(&input_outlet.0).unwrap() {
                n
            } else {
                panic!("Unsupported node type");
            };

            // For all node inputs, if the input's out_dims doesn't match the current node's output dims,
            // we insert a broadcast node in between.
            if input.out_dims != self.out_dims {
                let new_node_index = self.idx;
                let opkind = SupportedOp::Linear(PolyOp::MultiBroadcastTo {
                    shape: self.out_dims.clone(),
                });

                let b_node = Node {
                    idx: new_node_index,
                    opkind,
                    inputs: vec![*input_outlet],
                    out_dims: self.out_dims.clone(),
                    num_uses: 1,
                    out_scale: input.out_scale,
                };

                other_nodes.insert(new_node_index, NodeType::Node(b_node));
                self.inputs[j] = (new_node_index, 0);
                self.idx += 1;
            }
        }
    }
}

/// Maps outlets of the [`tract_onnx::prelude::Graph`] to nodes of the [`BTreeMap`] using the remappings.
///
/// # Arguments
///
/// * `outlets` - A slice of outlet IDs to map.
/// * `remappings` - A `BTreeMap` of remapping indices to apply. The keys are the `Graph` node indices,
///   and the values are the node indices in the `nodes` collection.
///   It is populated each time a new node from the `Graph` is added to the `nodes` collection.
pub fn map_outlet_indices(
    outlets: &[OutletId],
    remappings: &BTreeMap<usize, usize>,
) -> Vec<(usize, usize)> {
    outlets
        .iter()
        .map(|i| {
            let mapped_input = remappings.get(&i.node).unwrap_or_else(|| {
                panic!("Remapping for node {} not found", i.node);
            });
            (*mapped_input, i.slot)
        })
        .collect::<Vec<_>>()
}

impl Node {
    /// Decodes the current [Node] into an [ONNXInstr] at the specified `address`.
    ///
    /// This method is typically used during preprocessing to transform the ONNX binary into the zkVM program code.
    ///
    /// # Arguments
    ///
    /// * `address` - The memory address or program counter where the decoded instruction will be placed.
    ///
    /// # Returns
    ///
    /// An [ONNXInstr] representing the decoded instruction for this node.
    ///
    /// # Panics
    ///
    /// This method will panic if there is an unsupported operator
    pub fn decode(&self, address: usize) -> ONNXInstr {
        self.decode_with_opcode(&self.opkind, address)
    }

    /// Helper function to decode the node with a specific opcode.
    ///
    /// # Arguments
    /// * `op` - The operation to decode.
    /// * `address` - The address in the bytecode where this instruction will be placed.
    ///
    /// # Returns
    /// An [ONNXInstr] representing the decoded instruction.
    ///
    /// # Panics
    /// Panics if the operation does not have exactly two operands, as this is expected for the current implementation.
    fn decode_with_opcode<T>(&self, op: &T, address: usize) -> ONNXInstr
    where
        for<'a> &'a T: Into<ONNXOpcode> + Debug,
    {
        // Helper fn to extract the node-idx from the Outlet type
        let node_idx = |(idx, _): &(usize, usize)| *idx;
        ONNXInstr {
            address,
            opcode: op.into(),
            ts1: self.inputs.first().map(node_idx),
            ts2: self.inputs.get(1).map(node_idx),
            ts3: self.inputs.get(2).map(node_idx),
            // The output tensor is always the current node's index.
            td: Some(self.idx),
            imm: self.imm(),
            virtual_sequence_remaining: None,
            active_output_elements: self.out_dims.iter().product(),
            output_dims: if self.out_dims.len() == 1 {
                [1, self.out_dims[0]]
            } else {
                [self.out_dims[0], self.out_dims[1]]
            },
        }
    }

    fn imm(&self) -> Option<Tensor<i32>> {
        match &self.opkind {
            SupportedOp::Constant(constant) => Some(constant.quantized_values.clone()),
            SupportedOp::Nonlinear(LookupOp::Div { denom }) => {
                Some(Tensor::from((0..MAX_TENSOR_SIZE).map(|_| denom.0 as i32)))
            }
            // Provide immediate indices for Gather with constant_idx so downstream
            // address computation can derive gather read addresses correctly.
            SupportedOp::Hybrid(HybridOp::Gather {
                constant_idx: Some(idx),
                ..
            }) => {
                let idx_i32 = idx.map(|x| x as i32);
                Some(idx_i32)
            }
            _ => None,
        }
    }
}

/// A single operation in a [crate::graph::Model].

#[derive(Clone, Debug, PartialEq)]
pub enum SupportedOp {
    /// A linear operation.
    Linear(PolyOp<i32>),
    /// A nonlinear operation.
    Nonlinear(LookupOp),
    /// A hybrid operation.
    Hybrid(HybridOp),
    /// An input node (e.g., model input or placeholder).
    Input(Input),
    /// A constant value node (e.g., weights, biases, or fixed tensors).
    Constant(Constant<i32>),
    /// An unknown or unsupported operation.
    Unknown(Unknown),
    /// An operation whose inputs have been rescaled for homogeneity.
    Rescaled(Rescaled),
    /// An operation whose output scale has been rebased to match the global scale.
    RebaseScale(RebaseScale),
}

impl Default for SupportedOp {
    fn default() -> Self {
        SupportedOp::Unknown(Unknown)
    }
}

impl From<&SupportedOp> for ONNXOpcode {
    fn from(op: &SupportedOp) -> Self {
        match op {
            SupportedOp::Linear(poly_op) => poly_op.into(),
            SupportedOp::Nonlinear(lookup_op) => lookup_op.into(),
            SupportedOp::Hybrid(hybrid_op) => hybrid_op.into(),
            SupportedOp::Input(input_op) => input_op.into(),
            SupportedOp::Constant(constant) => constant.into(),
            SupportedOp::RebaseScale(rebase_scale) => rebase_scale.into(),
            SupportedOp::Unknown(unknown) => unknown.into(),
            SupportedOp::Rescaled(rescaled) => (&*rescaled.inner).into(),
        }
    }
}

impl SupportedOp {
    ///
    pub fn is_lookup(&self) -> bool {
        match self {
            SupportedOp::Nonlinear(_) => true,
            SupportedOp::RebaseScale(op) => op.inner.is_lookup(),
            _ => false,
        }
    }
    ///
    pub fn get_input(&self) -> Option<Input> {
        match self {
            SupportedOp::Input(op) => Some(op.clone()),
            _ => None,
        }
    }

    ///
    pub fn get_rebased(&self) -> Option<&RebaseScale> {
        match self {
            SupportedOp::RebaseScale(op) => Some(op),
            _ => None,
        }
    }

    ///
    pub fn get_lookup(&self) -> Option<&LookupOp> {
        match self {
            SupportedOp::Nonlinear(op) => Some(op),
            _ => None,
        }
    }

    ///
    pub fn get_constant(&self) -> Option<&Constant<i32>> {
        match self {
            SupportedOp::Constant(op) => Some(op),
            _ => None,
        }
    }

    ///
    pub fn get_mutable_constant(&mut self) -> Option<&mut Constant<i32>> {
        match self {
            SupportedOp::Constant(op) => Some(op),
            _ => None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn homogenous_rescale(
        &self,
        in_scales: Vec<crate::Scale>,
    ) -> Result<Box<dyn Op<i32>>, Box<dyn Error>> {
        use crate::graph::utilities::homogenize_input_scales;

        let inputs_to_scale = self.requires_homogenous_input_scales();
        // creates a rescaled op if the inputs are not homogenous
        let op = self.clone_dyn();
        homogenize_input_scales(op, in_scales, inputs_to_scale)
    }

    /// Since each associated value of `SupportedOp` implements `Op`, let's define a
    /// helper method to retrieve it.
    fn as_op(&self) -> &dyn Op<i32> {
        match self {
            SupportedOp::Linear(op) => op,
            SupportedOp::Nonlinear(op) => op,
            SupportedOp::Hybrid(op) => op,
            SupportedOp::Input(op) => op,
            SupportedOp::Constant(op) => op,
            SupportedOp::Unknown(op) => op,
            SupportedOp::Rescaled(op) => op,
            SupportedOp::RebaseScale(op) => op,
        }
    }

    pub fn gen_node(&self, inputs: Vec<Outlet>, out_dims: Vec<usize>, idx: usize) -> Node {
        match self {
            SupportedOp::Input(op) => Node {
                opkind: self.clone(),
                out_scale: <Input as Op<i32>>::out_scale(op, vec![]).unwrap(),
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
            SupportedOp::Linear(op) => Node {
                opkind: self.clone(),
                out_scale: op.out_scale(vec![1, 1]).unwrap(),
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
            SupportedOp::Constant(op) => Node {
                opkind: self.clone(),
                out_scale: op.out_scale(vec![1]).unwrap(),
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
            SupportedOp::Nonlinear(op) => Node {
                opkind: self.clone(),
                out_scale: <LookupOp as Op<i32>>::out_scale(op, vec![1]).unwrap(),
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
            SupportedOp::Hybrid(op) => Node {
                opkind: self.clone(),
                out_scale: <HybridOp as Op<i32>>::out_scale(op, vec![1]).unwrap(),
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
            SupportedOp::Unknown(_) => Node {
                opkind: self.clone(),
                out_scale: 0,
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
            SupportedOp::Rescaled(op) => Node {
                opkind: self.clone(),
                out_scale: <Rescaled as Op<i32>>::out_scale(op, vec![1]).unwrap(),
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
            SupportedOp::RebaseScale(op) => Node {
                opkind: self.clone(),
                out_scale: <RebaseScale as Op<i32>>::out_scale(op, vec![1]).unwrap(),
                inputs,
                out_dims,
                idx,
                num_uses: 1,
            },
        }
    }
}

impl From<Box<dyn Op<i32>>> for SupportedOp {
    fn from(value: Box<dyn Op<i32>>) -> Self {
        if let Some(op) = value.as_any().downcast_ref::<PolyOp<i32>>() {
            return SupportedOp::Linear(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<LookupOp>() {
            return SupportedOp::Nonlinear(*op);
        };

        if let Some(op) = value.as_any().downcast_ref::<HybridOp>() {
            return SupportedOp::Hybrid(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Input>() {
            return SupportedOp::Input(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Constant<i32>>() {
            return SupportedOp::Constant(op.clone());
        };

        if let Some(op) = value.as_any().downcast_ref::<Unknown>() {
            return SupportedOp::Unknown(op.clone());
        };
        if let Some(op) = value.as_any().downcast_ref::<Rescaled>() {
            return SupportedOp::Rescaled(op.clone());
        };
        if let Some(op) = value.as_any().downcast_ref::<RebaseScale>() {
            return SupportedOp::RebaseScale(op.clone());
        };

        log::error!("Unsupported op type");
        log::warn!("defaulting to Unknown");
        SupportedOp::Unknown(Unknown {})
    }
}

impl Op<i32> for SupportedOp {
    fn f(&self, inputs: &[Tensor<i32>]) -> Result<ForwardResult<i32>, crate::tensor::TensorError> {
        self.as_op().f(inputs)
    }

    fn is_input(&self) -> bool {
        self.as_op().is_input()
    }

    fn is_constant(&self) -> bool {
        self.as_op().is_constant()
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        self.as_op().requires_homogenous_input_scales()
    }

    fn requires_shape_equality(&self) -> bool {
        self.as_op().requires_shape_equality()
    }

    fn clone_dyn(&self) -> Box<dyn Op<i32>> {
        self.as_op().clone_dyn()
    }

    fn as_string(&self) -> String {
        self.as_op().as_string()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        self.as_op().required_lookups()
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        self.as_op().out_scale(in_scales)
    }
}

/// A wrapper for an operation that has been rescaled.
#[derive(Clone, Debug, PartialEq)]
pub struct Rescaled {
    /// The operation that has to be rescaled.
    pub inner: Box<SupportedOp>,
    /// The scale of the operation's inputs.
    pub scale: Vec<(usize, u128)>,
}

impl Op<i32> for Rescaled {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn f(&self, x: &[Tensor<i32>]) -> Result<ForwardResult<i32>, TensorError> {
        if self.scale.len() != x.len() {
            return Err(TensorError::DimMismatch("rescaled inputs".to_string()));
        }
        let mut rescaled_inputs = vec![];
        let inputs = &mut x.to_vec();
        for (i, ri) in inputs.iter_mut().enumerate() {
            let mult_tensor = Tensor::from([self.scale[i].1 as i32].into_iter());
            let res = (ri.clone() * mult_tensor)?;
            rescaled_inputs.push(res);
        }
        Op::<i32>::f(&*self.inner, &rescaled_inputs)
    }

    fn as_string(&self) -> String {
        format!("RESCALED INPUT ({})", self.inner.as_string())
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let in_scales = in_scales
            .into_iter()
            .zip(self.scale.iter())
            .map(|(a, b)| a + multiplier_to_scale(b.1 as f64))
            .collect();

        Op::<i32>::out_scale(&*self.inner, in_scales)
    }

    fn clone_dyn(&self) -> Box<dyn Op<i32>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }

    fn requires_shape_equality(&self) -> bool {
        self.inner.requires_shape_equality()
    }
}

/// A wrapper for an operation that has been rescaled.
#[derive(Clone, Debug, PartialEq)]
pub struct RebaseScale {
    /// The operation that has to be rescaled.
    pub inner: Box<SupportedOp>,
    /// the multiplier applied to the node output
    pub multiplier: f64,
    /// scale being rebased to
    pub target_scale: i32,
    /// The original scale of the operation's inputs.
    pub original_scale: i32,
}

impl RebaseScale {
    ///
    pub fn rebase(
        inner: SupportedOp,
        global_scale: crate::Scale,
        op_out_scale: crate::Scale,
        scale_rebase_multiplier: u32,
    ) -> SupportedOp {
        if (op_out_scale > (global_scale * scale_rebase_multiplier as i32))
            && !inner.is_constant()
            && !inner.is_input()
        {
            let multiplier =
                scale_to_multiplier(op_out_scale - global_scale * scale_rebase_multiplier as i32);
            if let Some(op) = inner.get_rebased() {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: op.inner.clone(),
                    target_scale: op.target_scale,
                    multiplier: op.multiplier * multiplier,
                    original_scale: op.original_scale,
                })
            } else {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: Box::new(inner),
                    target_scale: global_scale * scale_rebase_multiplier as i32,
                    multiplier,
                    original_scale: op_out_scale,
                })
            }
        } else {
            inner
        }
    }

    ///
    pub fn rebase_up(
        inner: SupportedOp,
        target_scale: crate::Scale,
        op_out_scale: crate::Scale,
    ) -> SupportedOp {
        if (op_out_scale < (target_scale)) && !inner.is_constant() && !inner.is_input() {
            let multiplier = scale_to_multiplier(op_out_scale - target_scale);
            if let Some(op) = inner.get_rebased() {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: op.inner.clone(),
                    target_scale: op.target_scale,
                    multiplier: op.multiplier * multiplier,
                    original_scale: op.original_scale,
                })
            } else {
                SupportedOp::RebaseScale(RebaseScale {
                    inner: Box::new(inner),
                    target_scale,
                    multiplier,
                    original_scale: op_out_scale,
                })
            }
        } else {
            inner
        }
    }
}

impl Op<i32> for RebaseScale {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn f(&self, x: &[Tensor<i32>]) -> Result<ForwardResult<i32>, TensorError> {
        let mut res = Op::<i32>::f(&*self.inner, x)?;
        let ri = res.output;
        let rescaled = crate::tensor::ops::nonlinearities::const_div(&ri, self.multiplier);
        res.output = rescaled;

        res.intermediate_lookups.push(ri);

        Ok(res)
    }

    fn as_string(&self) -> String {
        format!(
            "REBASED (div={:?}) ({})",
            self.multiplier,
            self.inner.as_string()
        )
    }

    fn out_scale(&self, _: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        Ok(self.target_scale)
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        let mut lookups = self.inner.required_lookups();
        lookups.push(LookupOp::Div {
            denom: crate::ops::utils::F32(self.multiplier as f32),
        });
        lookups
    }

    fn clone_dyn(&self) -> Box<dyn Op<i32>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }

    fn requires_shape_equality(&self) -> bool {
        self.inner.requires_shape_equality()
    }
}

impl From<&RebaseScale> for ONNXOpcode {
    fn from(value: &RebaseScale) -> Self {
        ONNXOpcode::RebaseScale(Box::new((&*value.inner).into()))
    }
}

impl Tabled for Node {
    const LENGTH: usize = 6;

    fn headers() -> Vec<std::borrow::Cow<'static, str>> {
        let mut headers = Vec::with_capacity(Self::LENGTH);
        for i in [
            "idx",
            "opkind",
            "out_scale",
            "inputs",
            "out_dims",
            "required_lookups",
        ] {
            headers.push(std::borrow::Cow::Borrowed(i));
        }
        headers
    }

    fn fields(&self) -> Vec<std::borrow::Cow<'_, str>> {
        let mut fields = Vec::with_capacity(Self::LENGTH);
        fields.push(std::borrow::Cow::Owned(self.idx.to_string()));
        fields.push(std::borrow::Cow::Owned(display_opkind(&self.opkind)));
        fields.push(std::borrow::Cow::Owned(self.out_scale.to_string()));
        fields.push(std::borrow::Cow::Owned(display_vector(&self.inputs)));
        fields.push(std::borrow::Cow::Owned(display_vector(&self.out_dims)));
        fields.push(std::borrow::Cow::Owned(format!(
            "{:?}",
            self.opkind
                .required_lookups()
                .iter()
                .map(<LookupOp as Op<i32>>::as_string)
                .collect_vec()
        )));
        fields
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        (self.out_scale == other.out_scale)
            && (self.inputs == other.inputs)
            && (self.out_dims == other.out_dims)
            && (self.idx == other.idx)
            && (self.opkind.as_string() == other.opkind.as_string())
    }
}

/// Rescales a constant node's quantized values in-place if it is only used once and its scale does not match the required input scale(s).
///
/// In quantized computation graphs, operations often require all their inputs to have the same fixed-point scale for correct arithmetic.
/// If a constant (such as a weight or bias tensor) is only used by a single node, it is safe and efficient to rescale it in-place to match the consumer's required scale.
/// This avoids inserting extra rescaling operations into the graph, reducing computational overhead and minimizing quantization error.
///
/// This function checks if the provided constant node is only used once (`num_uses == 1`). If so, it compares the constant's current output scale to the maximum required input scale among its consumers.
/// If the required scale is higher than the constant's current scale, it re-quantizes the constant's raw values to the new scale, updating its quantized representation in-place.
///
/// - If `num_uses == 1`, fetch the constant's current output scale.
/// - Determine the maximum required input scale from `in_scales`.
/// - If the required scale is greater than the current scale, re-quantize the constant's raw values to the new scale using `quantize_tensor`.
/// - Update the constant's quantized values in-place.
///
/// Use this function during graph construction or optimization passes, specifically when preparing input nodes for operations that require homogeneous input scales.
/// It is typically called as part of the node construction logic when building the computation graph from an ONNX model, just before inserting rescaling operations for constants.
///
/// # Arguments
/// - `constant`: The mutable reference to the constant node to potentially rescale.
/// - `in_scales`: The list of required input scales for the operation consuming this constant.
/// - `num_uses`: The number of times this constant node is used in the graph.
///
/// # Returns
/// Returns `Ok(())` if successful, or an error if scale information is missing or quantization fails.
///
/// # Example
/// ```ignore
/// // During node construction, for each constant input:
/// rescale_const_with_single_use(constant, input_scales, constant_node.num_uses())?;
/// ```
fn rescale_const_with_single_use(
    constant: &mut Constant<i32>,
    in_scales: Vec<crate::Scale>,
    num_uses: usize,
) -> Result<(), Box<dyn Error>> {
    if num_uses == 1 {
        let current_scale = constant.out_scale(vec![])?;
        let scale_max = in_scales.iter().max().ok_or("no scales")?;
        if scale_max > &current_scale {
            let raw_values = constant.raw_values.clone();
            constant.quantized_values = quantize_tensor(raw_values, *scale_max)?;
        }
    }
    Ok(())
}

fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    if !v.is_empty() {
        format!("{v:?}",)
    } else {
        String::new()
    }
}

fn display_opkind(v: &SupportedOp) -> String {
    v.as_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{model::NodeType, utilities::create_input_node},
        ops::poly::PolyOp,
    };
    use std::collections::BTreeMap;
    use tract_onnx::prelude::OutletId;

    /// Test the `map_outlet_indices` function with various remapping scenarios
    #[test]
    fn test_map_outlet_indices() {
        // Test case 1: No remappings
        let outlets = vec![
            OutletId::new(0, 0),
            OutletId::new(1, 0),
            OutletId::new(2, 1),
        ];

        // Graph:       [Node0, Node1, Node2]
        // No broadcasts added
        // node list:   [Node0, Node1, Node2]

        let mut remappings = BTreeMap::new(); // No remappings
        remappings.insert(0, 0);
        remappings.insert(1, 1);
        remappings.insert(2, 2); // Indexes are unchanged from Graph to nodes list

        let result = map_outlet_indices(&outlets, &remappings);
        assert_eq!(result, vec![(0, 0), (1, 0), (2, 1)]);

        // Test case 2: Simple remappings
        let outlets = vec![
            OutletId::new(0, 0),
            OutletId::new(1, 0),
            OutletId::new(2, 0),
        ];

        // Graph:       [Node0, Node1, Node2]
        // Add broadcast before nodes [0, 1]
        // node list:   [Broadcast, Node0, Broadcast, Node1, Node2]

        let mut remappings = BTreeMap::new();
        remappings.insert(0, 1); // Node at index 0 in Graph is now at index 1 in nodes list
        remappings.insert(1, 3); // Node at index 1 in Graph is now at index 3 in nodes list
        remappings.insert(2, 4); // Node at index 2 in Graph is now at index 4 in nodes list

        let result = map_outlet_indices(&outlets, &remappings);
        assert_eq!(result, vec![(1, 0), (3, 0), (4, 0)]); // Corrected expectations

        // Test case 3: Multiple remappings at same position
        let outlets = vec![
            OutletId::new(0, 0),
            OutletId::new(1, 0),
            OutletId::new(2, 0),
        ];

        // Graph:       [Node0, Node1, Node2]
        // Add broadcast before nodes [0, 0, 1]
        // node list:   [Broadcast, Broadcast, Node0, Broadcast, Node1, Node2]

        let mut remappings = BTreeMap::new();
        remappings.insert(0, 2); // Node at index 0 in Graph is now at index 2 in nodes list
        remappings.insert(1, 4); // Node at index 1 in Graph is now at index 4 in nodes list
        remappings.insert(2, 5); // Node at index 2 in Graph is now at index 5 in nodes list

        let result = map_outlet_indices(&outlets, &remappings);
        assert_eq!(result, vec![(2, 0), (4, 0), (5, 0)]);

        // Test case 4: Remappings beyond outlet indices
        let outlets = vec![OutletId::new(0, 0), OutletId::new(1, 1)];

        // Graph:       [Node0, Node1, Node2, Node3]
        // Add broadcast before nodes [0, 1, 2, 3]
        // node list:   [Broadcast, Node0, Broadcast, Node1, Broadcast, Node2, Broadcast, Node3]

        let mut remappings = BTreeMap::new();
        remappings.insert(0, 1); // Node at index 0 in Graph is now at index 1 in nodes list
        remappings.insert(1, 3); // Node at index 1 in Graph is now at index 3 in nodes list
        remappings.insert(2, 5); // Node at index 2 in Graph is now at index 5 in nodes list
        remappings.insert(3, 7); // Node at index 3 in Graph is now at index 7 in nodes list

        let result = map_outlet_indices(&outlets, &remappings);
        assert_eq!(result, vec![(1, 0), (3, 1)]);
    }

    /// Test the `homogenize_input_shapes` method for dimension matching
    #[test]
    fn test_homogenize_input_shapes_no_broadcast_needed() {
        let mut nodes = BTreeMap::new();

        // Create input node with dimensions [1, 3]
        let input_node = create_input_node(7, vec![1, 3], 0, 1);
        nodes.insert(0, NodeType::Node(input_node));

        // Create a node that expects [1, 3] input - no broadcasting needed
        let mut add_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0)],
            out_dims: vec![1, 3],
            num_uses: 1,
            out_scale: 7,
        };

        // No broadcast should be needed since dimensions match
        add_node.homogenize_input_shapes(&mut nodes);
        assert_eq!(add_node.inputs, vec![(0, 0)]); // Inputs unchanged
        assert_eq!(nodes.len(), 1); // No new nodes added
    }

    /// Test the `homogenize_input_shapes` method when broadcasting is needed
    #[test]
    fn test_homogenize_input_shapes_broadcast_needed() {
        let mut nodes = BTreeMap::new();

        // Create input node with dimensions [1] (scalar)
        let input_node = create_input_node(7, vec![1], 0, 1);
        nodes.insert(0, NodeType::Node(input_node));

        // Create a node that expects [1, 3] input - broadcasting needed
        let mut add_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0)],
            out_dims: vec![1, 3],
            num_uses: 1,
            out_scale: 7,
        };

        // Broadcast should be needed since [1] needs to become [1, 3]
        add_node.homogenize_input_shapes(&mut nodes);
        assert_eq!(add_node.inputs, vec![(1, 0)]); // Input now points to broadcast node
        assert_eq!(add_node.idx, 2); // Node index incremented
        assert_eq!(nodes.len(), 2); // New broadcast node added

        // Check the broadcast node was created correctly
        if let Some(NodeType::Node(broadcast_node)) = nodes.get(&1) {
            assert_eq!(broadcast_node.idx, 1);
            assert_eq!(broadcast_node.inputs, vec![(0, 0)]); // Points to original input
            assert_eq!(broadcast_node.out_dims, vec![1, 3]); // Broadcasted dimensions
            assert_eq!(broadcast_node.out_scale, 7); // Same scale as input
            assert_eq!(broadcast_node.num_uses, 1);
            match &broadcast_node.opkind {
                SupportedOp::Linear(PolyOp::MultiBroadcastTo { shape }) => {
                    assert_eq!(shape, &vec![1, 3]);
                }
                _ => panic!("Expected MultiBroadcastTo operation"),
            }
        } else {
            panic!("Broadcast node not found");
        }
    }

    /// Test multiple inputs with mixed broadcasting requirements
    #[test]
    fn test_homogenize_input_shapes_multiple_inputs() {
        let mut nodes = BTreeMap::new();

        // Create two input nodes with different dimensions
        let input_node1 = create_input_node(7, vec![1], 0, 1); // Scalar [1]
        let input_node2 = create_input_node(7, vec![1, 3], 1, 1); // Vector [1, 3]
        nodes.insert(0, NodeType::Node(input_node1));
        nodes.insert(1, NodeType::Node(input_node2));

        // Create a node that expects [1, 3] for both inputs
        let mut add_node = Node {
            idx: 2,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0), (1, 0)],
            out_dims: vec![1, 3],
            num_uses: 1,
            out_scale: 7,
        };

        // Only first input should need broadcasting
        add_node.homogenize_input_shapes(&mut nodes);
        assert_eq!(add_node.inputs, vec![(2, 0), (1, 0)]); // First input redirected to broadcast
        assert_eq!(add_node.idx, 3); // Node index incremented
        assert_eq!(nodes.len(), 3); // One broadcast node added

        // Check the broadcast node was created for first input only
        if let Some(NodeType::Node(broadcast_node)) = nodes.get(&2) {
            assert_eq!(broadcast_node.inputs, vec![(0, 0)]); // Points to first input
            assert_eq!(broadcast_node.out_dims, vec![1, 3]);
        } else {
            panic!("Broadcast node not found");
        }
    }

    /// Test with complex dimension mismatches
    #[test]
    fn test_homogenize_input_shapes_complex_dimensions() {
        let mut nodes = BTreeMap::new();

        // Create input node with dimensions [2]
        let input_node = create_input_node(7, vec![2], 0, 1);
        nodes.insert(0, NodeType::Node(input_node));

        // Create a node that expects [2, 4] input
        let mut mul_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::Mult),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 4],
            num_uses: 1,
            out_scale: 7,
        };

        // Broadcast should be needed to go from [2] to [2, 4]
        mul_node.homogenize_input_shapes(&mut nodes);
        assert_eq!(mul_node.inputs, vec![(1, 0)]); // Input now points to broadcast node
        assert_eq!(mul_node.idx, 2); // Node index incremented

        // Check broadcast node dimensions
        if let Some(NodeType::Node(broadcast_node)) = nodes.get(&1) {
            assert_eq!(broadcast_node.out_dims, vec![2, 4]);
            match &broadcast_node.opkind {
                SupportedOp::Linear(PolyOp::MultiBroadcastTo { shape }) => {
                    assert_eq!(shape, &vec![2, 4]);
                }
                _ => panic!("Expected MultiBroadcastTo operation"),
            }
        } else {
            panic!("Broadcast node not found");
        }
    }

    /// Test that no broadcasting occurs when dimensions already match exactly
    #[test]
    fn test_homogenize_input_shapes_exact_match() {
        let mut nodes = BTreeMap::new();

        // Create input nodes with matching dimensions
        let input_node1 = create_input_node(7, vec![2, 3], 0, 1);
        let input_node2 = create_input_node(7, vec![2, 3], 1, 1);
        nodes.insert(0, NodeType::Node(input_node1));
        nodes.insert(1, NodeType::Node(input_node2));

        // Create a node that expects [2, 3] - exact match
        let mut sub_node = Node {
            idx: 2,
            opkind: SupportedOp::Linear(PolyOp::Sub),
            inputs: vec![(0, 0), (1, 0)],
            out_dims: vec![2, 3],
            num_uses: 1,
            out_scale: 7,
        };

        // No broadcasting should be needed
        sub_node.homogenize_input_shapes(&mut nodes);
        assert_eq!(sub_node.inputs, vec![(0, 0), (1, 0)]); // Inputs unchanged
        assert_eq!(sub_node.idx, 2); // Index unchanged
        assert_eq!(nodes.len(), 2); // No new nodes
    }
}
