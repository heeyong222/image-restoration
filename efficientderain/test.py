from torchinfo import summary

import torch
from torchinfo import summary  # pip install torchinfo
from network import KPN  # KPN 클래스가 정의된 모듈 임포트포트

def print_model_summaries(pretrained_path="./efficientderain/v4_SPA.pth", device="cuda"):
    # 1. 순수 모델 (가중치 초기화된 모델)
    model_pure = KPN(
        color=True, 
        burst_length=1, 
        blind_est=True, 
        kernel_size=[3],
        sep_conv=False, 
        channel_att=False, 
        spatial_att=False,
        upMode='bilinear', 
        core_bias=False
    )
    model_pure.to(device)
    model_pure.eval()
    
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    print("========== Pure Model Summary ==========")
    summary(model_pure, input_data=[dummy_input, dummy_input], verbose=2)
    
    # 2. Pretrained 모델 (체크포인트 로드)
    # model_pretrained = KPN(
    #     color=True, 
    #     burst_length=1, 
    #     blind_est=True, 
    #     kernel_size=[5],
    #     sep_conv=False, 
    #     channel_att=False, 
    #     spatial_att=False,
    #     upMode='bilinear', 
    #     core_bias=False
    # )
    model_pretrained = KPN(color=True, burst_length=1, blind_est=True, kernel_size=[3],
                       sep_conv=False, channel_att=False, spatial_att=False,
                       upMode='bilinear', core_bias=False)
    # pretrained weight 로드 (체크포인트의 모델 아키텍처가 동일하다고 가정)
    state_dict = torch.load(pretrained_path, map_location=device)
    model_pretrained.load_state_dict(state_dict)
    model_pretrained.to(device)
    model_pretrained.eval()
    
    print("\n========== Pretrained Model Summary ==========")
    summary(model_pretrained, input_data=[dummy_input, dummy_input])

if __name__ == "__main__":
    print_model_summaries()