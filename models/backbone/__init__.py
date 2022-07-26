from .skip import skip


def get_net(
    input_depth,
    pad,
    upsample_mode,
    n_channels=3,
    act_fun="LeakyReLU",
    skip_n33d=128,
    skip_n33u=128,
    skip_n11=4,
    num_scales=5,
    downsample_mode="stride",
    need_sigmoid=True,
    need_tanh=False,
    decorr_rgb=False,
):
    assert need_sigmoid != need_tanh
    net = skip(
        input_depth,
        n_channels,
        num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d,
        num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u,
        num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11,
        upsample_mode=upsample_mode,
        downsample_mode=downsample_mode,
        need_sigmoid=need_sigmoid,
        need_tanh=need_tanh,
        need_bias=True,
        pad=pad,
        act_fun=act_fun,
        decorr_rgb=decorr_rgb,
    )
    return net
