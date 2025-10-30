import onnx
from onnx import helper, numpy_helper
import numpy as np

def replace_div_with_mul_reciprocal(model_path, output_path):
    """Transform Div operations to Mul by reciprocal"""
    model = onnx.load(model_path)
    graph = model.graph
    
    nodes_to_remove = []
    nodes_to_add = []
    
    for node in graph.node:
        if node.op_type == 'Div':
            print(f"Found Div node: {node.name}")
            divisor_name = node.input[1]
            
            # Find constant divisor
            constant_tensor = None
            for initializer in graph.initializer:
                if initializer.name == divisor_name:
                    constant_tensor = numpy_helper.to_array(initializer)
                    break
            
            if constant_tensor is not None:
                # Calculate reciprocal
                reciprocal = 1.0 / constant_tensor
                
                # Create reciprocal constant
                reciprocal_name = f"{divisor_name}_reciprocal"
                reciprocal_tensor = numpy_helper.from_array(
                    reciprocal.astype(np.float32),
                    name=reciprocal_name
                )
                graph.initializer.append(reciprocal_tensor)
                
                # Create Mul node
                mul_node = helper.make_node(
                    'Mul',
                    inputs=[node.input[0], reciprocal_name],
                    outputs=node.output,
                    name=node.name.replace('/Div', '/Mul_reciprocal')
                )
                
                nodes_to_remove.append(node)
                nodes_to_add.append(mul_node)
                print(f"  ✓ Replaced with Mul by reciprocal {reciprocal}")
            else:
                print(f"  ✗ Divisor is not constant, keeping Div")
    
    # Apply transformations
    for node in nodes_to_remove:
        graph.node.remove(node)
    for node in nodes_to_add:
        graph.node.append(node)
    
    # Save
    onnx.save(model, output_path)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Replaced {len(nodes_to_remove)} Div operations")
    return len(nodes_to_remove) > 0

# Transform percentage_limit
if __name__ == "__main__":
    transformed = replace_div_with_mul_reciprocal(
        "percentage_limit.onnx",
        "percentage_limit_no_div.onnx"
    )
    if transformed:
        print("\n✓ Model transformed successfully")
    else:
        print("\n✗ No Div operations found")
