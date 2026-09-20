# Usage: source ~/vla_finetune/activate_gr00t.sh
source "$HOME/vla_finetune/storage_env.sh"

export UV_PROJECT_ENVIRONMENT="$VLA_STORAGE_ROOT/envs/gr00t-n1.7"
export TMPDIR="$VLA_STORAGE_ROOT/tmp"
export CUDA_HOME=/usr/local/cuda-12.8
gr00t_runtime="$HOME/raid/conda/envs/gr00t-runtime"
case ":$PATH:" in
    *":$gr00t_runtime/bin:"*) ;;
    *) export PATH="$gr00t_runtime/bin:$PATH" ;;
esac
case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac
# Expose only FFmpeg libraries. The full conda lib directory can override
# the system libtinfo and break /usr/bin/tmux after activation.
# Remove the old setting too, so sourcing this file repairs an active shell.
gr00t_runtime_real="$(readlink -f "$gr00t_runtime")"
gr00t_storage_real="$(readlink -f "$VLA_STORAGE_ROOT")"
gr00t_ld_clean=""
IFS=: read -r -a gr00t_ld_entries <<< "${LD_LIBRARY_PATH-}"
for gr00t_ld_entry in "${gr00t_ld_entries[@]}"; do
    case "$gr00t_ld_entry" in
        "$gr00t_runtime/lib"|"$gr00t_runtime_real/lib"|"$VLA_STORAGE_ROOT/ffmpeg-libs"|"$gr00t_storage_real/ffmpeg-libs"|"") ;;
        *) gr00t_ld_clean="${gr00t_ld_clean:+$gr00t_ld_clean:}$gr00t_ld_entry" ;;
    esac
done
export LD_LIBRARY_PATH="$VLA_STORAGE_ROOT/ffmpeg-libs${gr00t_ld_clean:+:$gr00t_ld_clean}"
unset gr00t_runtime gr00t_runtime_real gr00t_storage_real
unset gr00t_ld_clean gr00t_ld_entries gr00t_ld_entry

if [ ! -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then
    printf 'GR00T environment is not installed yet: %s\n' "$UV_PROJECT_ENVIRONMENT" >&2
    return 1 2>/dev/null || exit 1
fi
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
