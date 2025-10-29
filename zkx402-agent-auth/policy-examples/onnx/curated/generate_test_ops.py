#!/usr/bin/env python3
"""
Generate test ONNX models for Less, Slice, Identity, and Clip operations
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnx.checker

def create_less_test():
    """Test Less operation: value < threshold"""
    # Inputs
    value = helper.make_tensor_value_info('value', TensorProto.INT32, [1])
    threshold = helper.make_tensor_value_info('threshold', TensorProto.INT32, [1])

    # Output
    result = helper.make_tensor_value_info('result', TensorProto.BOOL, [1])

    # Less node
    less_node = helper.make_node(
        'Less',
        inputs=['value', 'threshold'],
        outputs=['result']
    )

    # Graph
    graph = helper.make_graph(
        [less_node],
        'less_test',
        [value, threshold],
        [result]
    )

    # Model
    model = helper.make_model(graph, producer_name='zkx402-test')
    model.opset_import[0].version = 13

    onnx.checker.check_model(model)
    onnx.save(model, 'test_less.onnx')
    print("✅ Created test_less.onnx")
    return model

def create_identity_test():
    """Test Identity operation: pass-through"""
    # Input
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [1])

    # Output
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [1])

    # Identity node
    identity_node = helper.make_node(
        'Identity',
        inputs=['x'],
        outputs=['y']
    )

    # Graph
    graph = helper.make_graph(
        [identity_node],
        'identity_test',
        [x],
        [y]
    )

    # Model
    model = helper.make_model(graph, producer_name='zkx402-test')
    model.opset_import[0].version = 13

    onnx.checker.check_model(model)
    onnx.save(model, 'test_identity.onnx')
    print("✅ Created test_identity.onnx")
    return model

def create_clip_test():
    """Test Clip operation: clamp values to [min, max]"""
    # Input
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [1])

    # Output
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [1])

    # Min and max constants
    min_val = helper.make_tensor('min', TensorProto.INT32, [], [0])
    max_val = helper.make_tensor('max', TensorProto.INT32, [], [100])

    # Clip node
    clip_node = helper.make_node(
        'Clip',
        inputs=['x', 'min', 'max'],
        outputs=['y']
    )

    # Graph
    graph = helper.make_graph(
        [clip_node],
        'clip_test',
        [x],
        [y],
        initializer=[min_val, max_val]
    )

    # Model
    model = helper.make_model(graph, producer_name='zkx402-test')
    model.opset_import[0].version = 13

    onnx.checker.check_model(model)
    onnx.save(model, 'test_clip.onnx')
    print("✅ Created test_clip.onnx")
    return model

def create_slice_test():
    """Test Slice operation: extract subset of tensor"""
    # Input: 5-element array
    x = helper.make_tensor_value_info('x', TensorProto.INT32, [5])

    # Output: 3 elements (indices 1:4)
    y = helper.make_tensor_value_info('y', TensorProto.INT32, [3])

    # Slice parameters (start=1, end=4, axis=0, step=1)
    starts = helper.make_tensor('starts', TensorProto.INT64, [1], [1])
    ends = helper.make_tensor('ends', TensorProto.INT64, [1], [4])
    axes = helper.make_tensor('axes', TensorProto.INT64, [1], [0])
    steps = helper.make_tensor('steps', TensorProto.INT64, [1], [1])

    # Slice node
    slice_node = helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes', 'steps'],
        outputs=['y']
    )

    # Graph
    graph = helper.make_graph(
        [slice_node],
        'slice_test',
        [x],
        [y],
        initializer=[starts, ends, axes, steps]
    )

    # Model
    model = helper.make_model(graph, producer_name='zkx402-test')
    model.opset_import[0].version = 13

    onnx.checker.check_model(model)
    onnx.save(model, 'test_slice.onnx')
    print("✅ Created test_slice.onnx")
    return model

if __name__ == '__main__':
    print("Generating test ONNX models...")
    print()

    create_less_test()
    create_identity_test()
    create_clip_test()
    create_slice_test()

    print()
    print("All test models created successfully!")
    print()
    print("Test with:")
    print("  python3 -m onnxruntime test_less.onnx")
    print("  python3 -m onnxruntime test_identity.onnx")
    print("  python3 -m onnxruntime test_clip.onnx")
    print("  python3 -m onnxruntime test_slice.onnx")
