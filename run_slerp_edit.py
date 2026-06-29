"""
Single-image text-guided editing with the SlerpFlow sampler.

Method:
  `slerp` (flux/sampling.py) extends FireFlow's velocity-reuse sampler by
  performing a *spatial-local spherical interpolation* (Slerp) between the
  start-point velocity `v_curr` and the predicted end-point velocity `v_next`
  at every step, controlled by `slerp_t`, with magnitude aligned to `||v_next||`.
  This yields stronger, more semantically-accurate edits than plain FireFlow.

Pipeline (per image):
  inversion (guidance=1, inverse=True) -> denoise w/ target prompt
  (guidance, inverse=False) -> decode.

Model weights:
  Set env vars FLUX_DEV and AE to local .safetensors paths, otherwise the
  FLUX.1-dev weights are downloaded from HuggingFace (black-forest-labs/FLUX.1-dev).

Example:
  python run_slerp_edit.py \
      --source_img cat.jpg \
      --source_prompt "a cat sitting on a sofa" \
      --target_prompt "a tiger sitting on a sofa" \
      --output out.jpg \
      --strategy slerp --num_steps 15 --guidance 3.0 --inject 2 --slerp_t 0.5
"""
import argparse
import os
import sys

import numpy as np
import torch
from einops import rearrange
from PIL import Image

# allow running from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux.sampling import (
    slerp,
    denoise_fireflow,
    get_schedule,
    prepare,
    unpack,
)
from flux.util import load_ae, load_clip, load_flow_model, load_t5

STRATEGY_FUNCS = {
    "slerp": slerp,   # the method
    "fireflow": denoise_fireflow,  # baseline
}


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0).to(torch_device)
    return ae.encode(init_image).to(torch.bfloat16)


@torch.inference_mode()
def run_edit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(">>> loading models ...")
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(args.name, device="cpu" if args.offload else device)
    ae = load_ae(args.name, device="cpu" if args.offload else device)

    # read & crop image to a multiple of 16
    arr = np.array(Image.open(args.source_img).convert("RGB"))
    h0, w0 = arr.shape[:2]
    arr = arr[: h0 - h0 % 16, : w0 - w0 % 16, :]
    width, height = arr.shape[0], arr.shape[1]

    if args.offload:
        ae.encoder.to(device); t5.to(device); clip.to(device)
    encoded = encode(arr, device, ae)
    inp = prepare(t5, clip, encoded, prompt=args.source_prompt)
    inp_target = prepare(t5, clip, encoded, prompt=args.target_prompt)
    if args.offload:
        ae.encoder.cpu(); t5.cpu(); clip.cpu(); torch.cuda.empty_cache()
        model.to(device)

    info = {
        "feature": {},          # in-memory feature cache used for injection (layers.py)
        "inject_step": args.inject,
        "start_layer_index": 0,
        "end_layer_index": 37,
        "reuse_v": 1,
        "editing_strategy": "replace_v",
        "qkv_ratio": [1.0, 1.0, 1.0],
        "slerp_t": args.slerp_t,
    }

    strat_fn = STRATEGY_FUNCS[args.strategy]
    timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=(args.name != "flux-schnell"))

    # inversion: source image -> latent noise
    z, info = strat_fn(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
    inp_target["img"] = z

    # denoise with target prompt
    timesteps = get_schedule(args.num_steps, inp_target["img"].shape[1], shift=(args.name != "flux-schnell"))
    x, _ = strat_fn(model, **inp_target, timesteps=timesteps, guidance=args.guidance, inverse=False, info=info)

    if args.offload:
        model.cpu(); torch.cuda.empty_cache(); ae.decoder.to(device)

    x = unpack(x.float(), width, height)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x = ae.decode(x[0].unsqueeze(0))
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    out = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.save(args.output, quality=95, subsampling=0)
    print(f">>> saved: {args.output}")


def build_parser():
    p = argparse.ArgumentParser(description="SlerpFlow single-image editing")
    p.add_argument("--source_img", required=True, help="path to source image")
    p.add_argument("--source_prompt", required=True, help="prompt describing the source image")
    p.add_argument("--target_prompt", required=True, help="prompt describing the desired edit")
    p.add_argument("--output", default="output.jpg", help="output image path")
    p.add_argument("--name", default="flux-dev", choices=["flux-dev", "flux-schnell"])
    p.add_argument("--strategy", default="slerp", choices=list(STRATEGY_FUNCS))
    p.add_argument("--num_steps", type=int, default=15)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--inject", type=int, default=2, help="number of feature-injection steps")
    p.add_argument("--slerp_t", type=float, default=0.5, help="slerp interpolation factor (0..1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offload", action="store_true", help="offload modules to CPU to save VRAM")
    return p


if __name__ == "__main__":
    run_edit(build_parser().parse_args())
