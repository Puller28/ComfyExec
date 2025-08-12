import base64, io, os
from typing import Dict, Any
from PIL import Image, ImageOps

import torch
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
import runpod

# ---------- Load model once ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SDXL = True  # flip to False if VRAM is tight

pipe_xl = None
pipe_15 = None

def get_pipe():
    global pipe_xl, pipe_15
    if USE_SDXL:
        if pipe_xl is None:
            # SDXL inpainting model (good quality)
            pipe_xl = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(DEVICE)
            pipe_xl.enable_attention_slicing()
        return pipe_xl
    else:
        if pipe_15 is None:
            # SD 1.5 inpainting model (cheaper VRAM)
            pipe_15 = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16
            ).to(DEVICE)
            pipe_15.enable_attention_slicing()
        return pipe_15

# ---------- Utils ----------
def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

def image_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def prepare_mask(mask_rgba: Image.Image, size, invert: bool) -> Image.Image:
    # convert to single-channel L mask 0..255 (white=fill, black=keep)
    if "A" in mask_rgba.getbands():
        m = mask_rgba.getchannel("A")
    else:
        m = mask_rgba.convert("L")
    if invert:
        m = ImageOps.invert(m)
    # ensure expected resolution for diffusers call
    if m.size != size:
        m = m.resize(size, Image.LANCZOS)
    return m

def prepare_image(img: Image.Image, size) -> Image.Image:
    # ensure RGB and same size as mask
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img.convert("RGB")

# ---------- Handler ----------
def handler(event: Dict[str, Any]):
    inp = event.get("input") or {}
    prompt       = inp.get("prompt", "high quality interior background, realistic lighting")
    image_b64    = inp["image_b64"]
    mask_b64     = inp["mask_b64"]
    steps        = int(inp.get("steps", 20))
    cfg          = float(inp.get("cfg", 7.5))
    seed         = inp.get("seed", None)
    invert_mask  = bool(inp.get("invert_mask", True))  # our rembg alpha usually has subject opaque => we inpaint background

    g = None
    if seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # decode
    img_rgba  = b64_to_image(image_b64)
    mask_rgba = b64_to_image(mask_b64)

    # choose a working size (keep within VRAM)
    target_size = img_rgba.size
    # You can clamp e.g. to 1024 here if you like:
    # target_size = (min(img_rgba.width, 1024), int(img_rgba.height * min(1024/img_rgba.width, 1)))

    mask_l = prepare_mask(mask_rgba, target_size, invert_mask)
    img_rgb = prepare_image(img_rgba, target_size)

    pipe = get_pipe()

    # SDXL signature
    if isinstance(pipe, StableDiffusionXLInpaintPipeline):
        out = pipe(
            prompt=prompt,
            image=img_rgb,
            mask_image=mask_l,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=g,
        )
        result = out.images[0]
    else:
        # SD 1.5 signature
        out = pipe(
            prompt=prompt,
            image=img_rgb,
            mask_image=mask_l,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=g,
        )
        result = out.images[0]

    return {"images": [image_to_b64(result, "PNG")]}

runpod.serverless.start({"handler": handler})
