import onnx
import os

print("=== CHECKING ALL MODELS FOR DIV OPERATIONS ===\n")

models_with_div = []
models_without_div = []

for filename in sorted(os.listdir('.')):
    if filename.endswith('.onnx') and not filename.endswith('_no_div.onnx'):
        try:
            model = onnx.load(filename)
            has_div = any(node.op_type == 'Div' for node in model.graph.node)
            
            if has_div:
                models_with_div.append(filename)
                print(f"❌ {filename:30s} - Uses Div")
            else:
                models_without_div.append(filename)
                print(f"✅ {filename:30s} - No Div")
        except Exception as e:
            print(f"⚠️  {filename:30s} - Error: {e}")

print(f"\n=== SUMMARY ===")
print(f"Models WITH Div:    {len(models_with_div)}")
print(f"Models WITHOUT Div: {len(models_without_div)}")

if models_with_div:
    print(f"\nModels that need division-free versions:")
    for model in models_with_div:
        print(f"  - {model}")
