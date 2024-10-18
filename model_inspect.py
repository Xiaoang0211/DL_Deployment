import tensorrt as trt

def load_engine(engine_file_path):
    # Create a logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Load the engine file into memory
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

if __name__ == "__main__":
    # Path to the .engine file
    engine_path = "/home/xiaoang/YOLOModels/yolov11/segmentation/yolo11s-seg.engine"

    # Load the engine
    engine = load_engine(engine_path)
    output_name = engine.get_tensor_name(1)
    output_shape_images = engine.get_tensor_shape("images")
    output_shape_images = engine.get_tensor_format_desc("output0")
    output_shape_output0 = engine.get_tensor_shape("output1")
    print(output_name)
    print(output_shape_images)
    print(output_shape_output0)