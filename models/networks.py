from torch.optim import lr_scheduler
from models.backbone.skip import skip


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def define_G(cfg):
    netG = skip(
        3,
        4,
        num_channels_down=[cfg["skip_n33d"]] * cfg["num_scales"]
        if isinstance(cfg["skip_n33d"], int)
        else cfg["skip_n33d"],
        num_channels_up=[cfg["skip_n33u"]] * cfg["num_scales"]
        if isinstance(cfg["skip_n33u"], int)
        else cfg["skip_n33u"],
        num_channels_skip=[cfg["skip_n11"]] * cfg["num_scales"]
        if isinstance(cfg["skip_n11"], int)
        else cfg["skip_n11"],
        need_bias=True,
    )
    return netG
