* main.cpp: Tensorrt inference with camera inference added, modified from [depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt)
* dynamo.py: Onnx inference with camera inference added, modified from [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX/tree/main)

## Demo of ViTL model
![ViTL Demo](./demo/vitl_demo.gif)
As we can see from ViTL Demo, the depth prediction is not consistent with the hand moving.
In the next step we are going to deploy [consistent_depth](https://github.com/facebookresearch/consistent_depth) that could potentially sove the inconsistency problem.
