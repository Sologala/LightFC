import torch
import torch.nn as nn

from lib.models import LightFC
from lib.test.utils import TrackerParams
from lib.utils.load import load_yaml


params = TrackerParams()
yaml_file = "experiments/lightfc/mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou.yaml"
params.cfg = load_yaml(yaml_file)
params.template_factor = 2.0
params.template_size = 128
params.search_factor = 4.0
params.search_size = 256
params.checkpoint = "./models/official/lightfc/checkpoints/train/lightfc/mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou/lightfc_ep0400.pth.tar"



class Backbone(nn.Module):
    def __init__(self, params):
        super(Backbone, self).__init__()
        self.net = LightFC(cfg=params.cfg, env_num=None, training=False)
        self.net.load_state_dict(torch.load(params.checkpoint, map_location="cpu")["net"], strict=True)

    def forward(self, z):
        return self.net.forward_backbone(z)
       
class Detection(nn.Module):
    def __init__(self, params):
        super(Detection, self).__init__()
        self.net = LightFC(cfg=params.cfg, env_num=None, training=False)
        self.net.load_state_dict(torch.load(params.checkpoint, map_location="cpu")["net"], strict=True)

    def forward(self, z, x):
        return self.net.forward_tracking_export_onnx(z, x)


def export_backone(params, export_onnx_path):
    backbone = Backbone(params)
    print(backbone)
    backbone.eval()

    # 创建随机输入张量
    dummy_input = torch.randn([1, 3, 128, 128])

    input_names = ["img"]
    output_names = ["feat"]

    torch.onnx.export(
        backbone,
        dummy_input,
        export_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names
    )


def export_detection(params, export_onnx_path):
    model = Detection(params)
    print(model)
    model.eval()

    # 创建随机输入张量
    dummy_x = torch.randn([1, 3, 256, 256])
    dummy_z = torch.randn([1, 96, 8, 8])

    input_names = ["img", "z"]
    output_names = ["score_map", "size_map", "offset_map"]

    torch.onnx.export(
        model,
        (dummy_z, dummy_x),
        export_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names
    )



export_backone(params, "./output/lightfc_backbone.onnx")
import os
os.system("onnxsim ./output/lightfc_backbone.onnx ./output/lightfc_backbone.onnx")
export_detection(params, "./output/lightfc_detection.onnx")

os.system("onnxsim ./output/lightfc_detection.onnx ./output/lightfc_detection.onnx")
