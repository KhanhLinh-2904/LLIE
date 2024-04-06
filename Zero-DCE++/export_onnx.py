import torch
import torch.nn as nn
import torch.nn.functional as F
import model

scale_factor = 12
torch_model = model.enhance_net_nopool(scale_factor)
torch_model.load_state_dict(torch.load('/home/user/low_light_enhancement/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth', map_location=torch.device('cpu')))
print(type(torch_model))
torch_input = torch.randn(1, 3, 3648, 5472)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("ZeroDCE++1.onnx")
# onnx_program.save("output.onnx")