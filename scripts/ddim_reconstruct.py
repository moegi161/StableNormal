import argparse
import csv
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from stablenormal.pipeline_stablenormal import StableNormalPipeline


def load_normal(path: str) -> np.ndarray:
    """Load a 16-bit normal map encoded in [0, 65535] to [-1, 1]."""
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr / 65535.0 * 2.0 - 1.0
    return arr


def tensor_from_normal(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def normals_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mean_angular_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8)
    target_n = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    dot = np.clip(np.sum(pred_n * target_n, axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return float(np.mean(ang))


def ddim_inversion(pipe: StableNormalPipeline, latents: torch.Tensor, num_steps: int, text_embed: torch.Tensor) -> torch.Tensor:
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(pipe.device)
    alphas = scheduler.alphas_cumprod.to(pipe.device)
    inv_timesteps = torch.flip(timesteps, dims=[0])
    x = latents
    for i, t in enumerate(inv_timesteps):
        latent_model_input = scheduler.scale_model_input(x, t)
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embed).sample
        alpha_t = alphas[t.long()]
        if i == len(inv_timesteps) - 1:
            alpha_next = torch.tensor(1.0, device=pipe.device)
        else:
            alpha_next = alphas[inv_timesteps[i + 1].long()]
        pred_x0 = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        x = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * noise_pred
    return x


def ddim_denoise(pipe: StableNormalPipeline, latents: torch.Tensor, num_steps: int, text_embed: torch.Tensor) -> torch.Tensor:
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.to(pipe.device)
    x = latents
    for i, t in enumerate(timesteps):
        latent_model_input = scheduler.scale_model_input(x, t)
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embed).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
    x = x / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(x).sample
    return image


def visualize_and_save(gt: np.ndarray, pred: np.ndarray, out_path: str) -> None:
    gt_img = normals_to_uint8(gt)
    pred_img = normals_to_uint8(pred)
    vis = np.concatenate([gt_img, pred_img], axis=1)
    Image.fromarray(vis).save(out_path)


def process_image(pipe: StableNormalPipeline, img_path: str, out_dir: str, steps_list: List[int], device: torch.device, prompts: List[str], results: List[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    gt = load_normal(img_path)
    base = Path(img_path).stem
    gt_tensor = tensor_from_normal(gt, device, pipe.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(gt_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
    uncond = pipe._encode_prompt("", device, 1, False)
    for steps in steps_list:
        latents_T = ddim_inversion(pipe, latents, steps, uncond)
        for prompt in prompts:
            text = pipe._encode_prompt(prompt, device, 1, False)
            with torch.no_grad():
                rec = ddim_denoise(pipe, latents_T.clone(), steps, text)
            rec_np = rec[0].permute(1, 2, 0).cpu().numpy()
            mse_val = mse(rec_np, gt)
            mae_val = mean_angular_error(rec_np, gt)
            label = "noprompt" if prompt == "" else "prompt"
            vis_path = os.path.join(out_dir, f"{base}_steps{steps}_{label}.png")
            visualize_and_save(gt, rec_np, vis_path)
            results.append({"image": base, "steps": steps, "prompt": prompt or "none", "mse": mse_val, "mae": mae_val})


def main() -> None:
    parser = argparse.ArgumentParser(description="DDIM inversion and reconstruction for normal maps")
    parser.add_argument("--input_dir", required=True, help="Directory with input tif normal maps")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--model_path", default="weights/stable-normal-v0-1", help="Path or repo id of StableNormal model")
    parser.add_argument("--timesteps", nargs="*", type=int, default=[10, 25, 50], help="Inference timesteps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableNormalPipeline.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    results: List[dict] = []
    img_paths = sorted([p for p in Path(args.input_dir).glob("*.tif")])
    prompts = ["", "a normal map"]
    for img_path in img_paths:
        img_out = os.path.join(args.output_dir, img_path.stem)
        process_image(pipe, str(img_path), img_out, args.timesteps, device, prompts, results)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "steps", "prompt", "mse", "mae"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    main()
