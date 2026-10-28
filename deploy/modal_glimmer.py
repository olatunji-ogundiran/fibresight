"""Serve Meta Muse Glimmer 30B (open weights, Apache 2.0) on Modal with vLLM.

  modal deploy deploy/modal_glimmer.py

The endpoint is OpenAI-compatible (chat completions with tool calls), so the
FibreSight agent talks to it exactly as it talks to Baseten or the Meta Model
API. Serving flags follow Meta's vLLM guide for Muse Glimmer: the model's own
tool-call and reasoning parsers, and its published sampling defaults.

Needs GLIMMER_API_KEY in the environment at deploy time; requests must send it
as a bearer token. The container scales to zero after a few idle minutes.
"""
import os
import subprocess

import modal

MODEL = "meta-models/Muse-Glimmer-30B"
VLLM_VERSION = "0.28.0"   # first release with the muse_glimmer tool-call and reasoning parsers
GPU = os.environ.get("GLIMMER_GPU", "H200")   # bf16 weights are about 60 GB
PORT = 8000

# vLLM 0.28 ships torch for CUDA 13.0, and FlashInfer compiles kernels at start-up, so the image needs
# the CUDA 13.0 toolkit (nvcc), not just the runtime.
image = (modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_xet]")
         .env({"CUDA_HOME": "/usr/local/cuda"}))
dl_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]")
weights = modal.Volume.from_name("muse-glimmer-weights", create_if_missing=True)
vllm_cache = modal.Volume.from_name("muse-glimmer-vllm-cache", create_if_missing=True)
jit_cache = modal.Volume.from_name("muse-glimmer-flashinfer-cache", create_if_missing=True)
secret = modal.Secret.from_dict({"GLIMMER_API_KEY": os.environ.get("GLIMMER_API_KEY", "")})

app = modal.App("fibresight-glimmer")


@app.function(image=dl_image, volumes={"/root/.cache/huggingface": weights}, timeout=60 * 60)
def download():
    """Fetch the weights into the volume on a CPU container (no GPU time spent)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL)
    weights.commit()
    print("weights at", path)


@app.function(image=image, gpu=GPU, timeout=60 * 60, scaledown_window=5 * 60,
              volumes={"/root/.cache/huggingface": weights, "/root/.cache/vllm": vllm_cache,
                       "/root/.cache/flashinfer": jit_cache},
              secrets=[secret], max_containers=1)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=25 * 60)
def serve():
    key = os.environ["GLIMMER_API_KEY"]
    assert key, "deploy with GLIMMER_API_KEY set"
    cmd = ["vllm", "serve", MODEL,
           "--served-model-name", "muse-glimmer",
           "--host", "0.0.0.0", "--port", str(PORT),
           "--tensor-parallel-size", "1",
           "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer",
           "--reasoning-parser", "muse_glimmer",
           "--generation-config", "auto",
           "--max-model-len", "131072",
           "--gpu-memory-utilization", "0.92",
           "--enable-prefix-caching",
           "--api-key", key]
    subprocess.Popen(cmd)
