import onnx

def inspect_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    # Get the model's graph
    graph = model.graph

    # Print input shapes
    print("Model Inputs:")
    for input_tensor in graph.input:
        name = input_tensor.name
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"Input Name: {name}, Shape: {shape}")

    # Print output shapes
    print("\nModel Outputs:")
    for output_tensor in graph.output:
        name = output_tensor.name
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"Output Name: {name}, Shape: {shape}")

if __name__ == "__main__":
    model_path = "/home/xiaoang/ultralytics/yolo11m-pose.onnx"  # Update this with the path to your ONNX model
    inspect_onnx_model(model_path)
