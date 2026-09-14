from .bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig


def load_ckpt(model, model_path, map_location="cpu"):
    if not model_path.endswith(".safetensors"):
        raise ValueError("Native ComfyUI nodes only accept safetensors VAE checkpoints")
    from safetensors.torch import load_file

    state_dict = load_file(model_path, device=map_location)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "AuK VAE checkpoint does not match its config: "
            f"missing weights={missing[:10]}, unexpected weights={unexpected[:10]}"
        )
    return model


def load_vae_model(vae_name, vae_cfg, vae_ckpt, **kwargs):
    if vae_name == "BigVGANFlowVAE":
        model = BigVGANFlowVAE(vae_cfg)
        assert vae_ckpt is not None, "BigVGANFlowVAE requires a checkpoint path"
        model = load_ckpt(model, vae_ckpt, **kwargs)
        model = model.eval()
        return model
    else:
        raise ValueError(f"Unknown VAE name: {vae_name}, Only support BigVGANFlowVAE")


__all__ = ["BigVGANFlowVAE", "BigVGANFlowVAEConfig", "load_vae_model"]
