import torch
import torch.nn.functional as F
import math
import lpips
import os

class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        # Initialize LPIPS model (AlexNet is standard)
        self.loss_fn_vgg = lpips.LPIPS(net='alex', version='0.1').to(device)
        self.loss_fn_vgg.eval()

    def psnr(self, pred, target, data_range=2.0):
        """
        Calculate PSNR. Images are assumed to be in range [-1, 1] so data_range is 2.0
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 10 * math.log10((data_range ** 2) / mse.item())

    def ssim(self, img1, img2, window_size=11, size_average=True):
        """
        Calculate SSIM.
        """
        # Convert from [-1, 1] to [0, 1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        
        channel = img1.size(1)
        
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        window = create_window(window_size, channel).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map.mean(1).mean(1).mean(1).item()

    def calc_lpips(self, pred, target):
        """
        Calculate LPIPS.
        LPIPS expects inputs in [-1, 1].
        """
        with torch.no_grad():
            dist = self.loss_fn_vgg(pred, target)
        return dist.mean().item()

    def evaluate_batch(self, pred, target):
        psnr_val = self.psnr(pred, target)
        ssim_val = self.ssim(pred, target)
        lpips_val = self.calc_lpips(pred, target)
        return psnr_val, ssim_val, lpips_val

# Note: FID is typically computed over entire directories using pytorch-fid:
# python -m pytorch_fid path/to/real path/to/generated
