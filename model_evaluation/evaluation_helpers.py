import torch
import torchvision
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as T

def KL_DivLoss(y_pred, y_true):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_input = F.log_softmax(y_pred, dim=1)
    log_target = F.log_softmax(y_true, dim=1)
    output = kl_loss(log_input, log_target)
    return output

def RMSELoss(y_pred, y_true):
    mse_loss = torch.nn.MSELoss(reduction="mean")
    output = torch.sqrt(mse_loss(y_true, y_pred))
    return output

def MAELoss(y_pred, y_true):
    mae_loss = torch.nn.L1Loss(reduction="mean")
    output = torch.sqrt(mae_loss(y_true, y_pred))
    return output  

def PSNR(y_pred, y_true):
    psnr = torchmetrics.PeakSignalNoiseRatio()
    output = psnr(y_pred, y_true)
    return output   

def SSIM(y_pred, y_true):
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()
    output = ssim(y_pred, y_true)
    return output

def FID(y_pred, y_true):
    from torchmetrics.image.fid import FrechetInceptionDistance
    
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid.update(y_true, real=True)
    fid.update(y_pred, real=False)
    output = fid.compute()
    return output

def calculate_metrics(y_pred, y_true):
    fn_list = [
        ("kl_div", KL_DivLoss), 
        ("rmse", RMSELoss), 
        ("mae", MAELoss),
        ("psnr", PSNR),
        ("ssim", SSIM),
        ("fid", FID)
    ]
    metric_dict = {}
    for fn_name, fn in fn_list:
        metric_dict[fn_name] = fn(y_pred, y_true)
    return metric_dict