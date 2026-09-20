# Shared cache locations for VLA work on this account.
# Source this file before starting Python, uv, or a model download.
# Hugging Face credentials keep their existing location in the private home.
export VLA_STORAGE_ROOT="$HOME/raid/vla_finetune"
export HF_HUB_CACHE="$HOME/raid/cache/huggingface/hub"
export HF_DATASETS_CACHE="$HOME/raid/cache/huggingface/datasets"
export HF_XET_CACHE="$HOME/raid/cache/huggingface/xet"
export HF_ASSETS_CACHE="$HOME/raid/cache/huggingface/assets"
export TORCH_HOME="$HOME/raid/cache/torch"
export TORCH_EXTENSIONS_DIR="$HOME/raid/cache/torch_extensions"
export TRITON_CACHE_DIR="$HOME/raid/cache/triton"
export CUDA_CACHE_PATH="$HOME/raid/cache/cuda"
export PIP_CACHE_DIR="$HOME/raid/cache/pip"
export UV_CACHE_DIR="$HOME/raid/cache/uv"
export UV_PYTHON_INSTALL_DIR="$HOME/raid/uv/python"
