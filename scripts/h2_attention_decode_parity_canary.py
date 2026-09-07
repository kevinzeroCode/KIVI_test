"""Stage H0-PRE tiny GPU production-parity canary.

Validates (does NOT scientifically collect) the decode-aware attention
distortion measurement design locked in docs/stage_h0_pre_registration.md:
reconstructed Key-only logits / Value-only output against DIRECTLY
observed production tensors, for one short real calibration prompt, one
layer, cached decode steps 1-4.

NOT scientific collection: no correlation computed, no A/B/C/D gate
evaluated, no 272-trajectory run. Writes only under
outputs/layer_attention_feature_canary/ -- never touches
outputs/layer_feature_pilot/ (frozen Stage-G data) or
outputs/layer_attention_feature_pilot/ (reserved for future real Stage-H
data).

Reuses production code directly wherever possible:
  - utils.attention_decode_features (this round's pure measurement math)
  - utils.layer_policy / utils.pilot_policy (policy construction)
  - utils.generation_semantics.resolve_generate_kwargs (real lcc generation
    kwargs, not reimplemented)
  - pred_long_bench.build_model_and_tokenizer / build_chat
  - models.llama_kivi.repeat_kv / cuda_bmm_fA_qB_outer (the REAL production
    kernels, called directly on captured production tensors -- never an
    independent reimplementation)

Usage:
    CUDA_VISIBLE_DEVICES=0 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
    ./.venv/bin/python scripts/h2_attention_decode_parity_canary.py
"""
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.attention_decode_features import (  # noqa: E402
    ShadowKVCache,
    aggregate_sample_decode_distortion,
    key_decode_distortion,
    parse_kivi_cache_tuple,
    value_decode_distortion,
)
from utils.layer_policy import resolve_layer_policy  # noqa: E402

DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs", "layer_attention_feature_canary")
CANARY_TASK = "lcc"
CANARY_DATASET_INDEX = 122  # shortest of the 4 preregistered lcc calibration samples (selection_length=650)
CANARY_LAYER = 0
CANARY_GROUP_SIZE = 32
CANARY_RESIDUAL_LENGTH = 128
CANARY_MAX_NEW_TOKENS = 6  # prefill(token#1) + decode steps 1..5 -> reaches the preferred step 4
CANARY_SEED = 42


class CanaryError(RuntimeError):
    pass


class _Args:
    pass


def get_boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def get_git_status_short():
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode()
    except Exception:
        return None


def gpu_snapshot():
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        procs = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return {"gpu": gpu, "compute_processes": procs}
    except Exception as e:
        return {"error": str(e)}


def tensor_diff_report(a, b, label):
    """Standard tensor-comparison protocol reused from Stage G2/G3 canaries:
    shape, dtype, max_abs_diff, mean_abs_diff, relative_l2, torch.equal,
    allclose. Never claims parity without reporting these explicitly."""
    import torch

    same_shape = list(a.shape) == list(b.shape)
    same_dtype = a.dtype == b.dtype
    result = {"label": label, "a_shape": list(a.shape), "b_shape": list(b.shape), "shape_equal": same_shape, "dtype_equal": same_dtype}
    if not same_shape:
        result.update({"torch_equal": False, "allclose_1e-3": False, "max_abs_diff": None, "mean_abs_diff": None, "relative_l2": None})
        return result
    diff = (a.float() - b.float()).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    diff_norm = torch.linalg.vector_norm((a.float() - b.float()).double())
    ref_norm = torch.clamp(torch.linalg.vector_norm(b.float().double()), min=1e-12)
    relative_l2 = float((diff_norm / ref_norm).item())
    result.update(
        {
            "torch_equal": bool(torch.equal(a, b)),
            "allclose_1e-3": bool(torch.allclose(a.float(), b.float(), atol=1e-3, rtol=1e-3)),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "relative_l2": relative_l2,
        }
    )
    return result


class LayerCallCapture:
    """One target layer's forward-call capture: post-RoPE Q/K (via a
    scoped spy on _apply_rotary_pos_emb_inplace), raw V (v_proj hook),
    the incoming past_key_value tuple (pre-hook on self_attn), the real
    pre-/post-softmax attention tensors (scoped softmax spy, decode calls
    only), and the real local attention output before o_proj (o_proj
    pre-hook). All spies are installed immediately before and removed
    immediately after the target layer's own forward() call, via a
    pre-hook/post-hook pair -- never active during any other layer's
    computation.
    """

    def __init__(self, attn_module, head_dim, num_kv_heads):
        self.attn_module = attn_module
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.calls = []  # list of dict, one per forward() invocation, in call order
        self._current = None
        self._real_rope_fn = None
        self._real_softmax = None
        self._handles = []

    def _install(self, module, args, kwargs):
        import models.llama_kivi as llama_kivi

        self._current = {
            "past_key_value_in": kwargs.get("past_key_value", None),
            "position_ids_in": kwargs.get("position_ids", None),
            "rotary_chunk_size": self.attn_module.rotary_chunk_size,
        }
        self._real_rope_fn = llama_kivi._apply_rotary_pos_emb_inplace
        self._real_softmax = llama_kivi.nn.functional.softmax

        current = self._current
        real_rope = self._real_rope_fn
        real_softmax = self._real_softmax

        def spy_rope(query_states, key_states, cos, sin, position_ids, chunk_size):
            q_out, k_out = real_rope(query_states, key_states, cos, sin, position_ids, chunk_size)
            current["q_post_rope"] = q_out.detach().clone()
            current["k_post_rope"] = k_out.detach().clone()
            return q_out, k_out

        def spy_softmax(input, dim=None, _stacklevel=3, dtype=None):
            result = real_softmax(input, dim=dim, dtype=dtype)
            # only the FIRST softmax call within this layer's own forward
            # matters (the decode branch calls softmax exactly once); keep
            # the first capture defensively in case of any future multi-call change.
            if "pre_softmax_logits" not in current:
                current["pre_softmax_logits"] = input.detach().clone()
                current["post_softmax_weights"] = result.detach().clone()
            return result

        llama_kivi._apply_rotary_pos_emb_inplace = spy_rope
        llama_kivi.nn.functional.softmax = spy_softmax

    def _uninstall(self, module, args, kwargs, output):
        import models.llama_kivi as llama_kivi

        llama_kivi._apply_rotary_pos_emb_inplace = self._real_rope_fn
        llama_kivi.nn.functional.softmax = self._real_softmax
        self.calls.append(self._current)
        self._current = None

    def _v_proj_hook(self, module, inp, output):
        self._current["v_proj_raw"] = output.detach().clone()

    def _q_proj_hook(self, module, inp, output):
        # Raw PRE-RoPE query -- captured independently of the RoPE spy, so
        # Q parity (Part 11) can be proven via a genuinely separate second
        # path, not merely re-observing the same transparent pass-through.
        self._current["q_proj_raw"] = output.detach().clone()

    def _rotary_emb_hook(self, module, inp, output):
        cos, sin = output
        self._current["cos"] = cos.detach().clone()
        self._current["sin"] = sin.detach().clone()

    def _o_proj_pre_hook(self, module, args, kwargs):
        self._current["attn_output_pre_o_proj"] = args[0].detach().clone()

    def install_handles(self):
        self._handles.append(self.attn_module.register_forward_pre_hook(self._install, with_kwargs=True))
        self._handles.append(self.attn_module.register_forward_hook(self._uninstall, with_kwargs=True))
        self._handles.append(self.attn_module.v_proj.register_forward_hook(self._v_proj_hook))
        self._handles.append(self.attn_module.q_proj.register_forward_hook(self._q_proj_hook))
        self._handles.append(self.attn_module.rotary_emb.register_forward_hook(self._rotary_emb_hook))
        self._handles.append(self.attn_module.o_proj.register_forward_pre_hook(self._o_proj_pre_hook, with_kwargs=True))

    def remove_handles(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reshape_v(self, bsz, q_len):
        raw = self.calls[-1]["v_proj_raw"]
        return raw.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()


def build_prompt(tokenizer, model_short_name, json_obj, task=CANARY_TASK):
    """`task` defaults to CANARY_TASK so every existing H2 call site (which
    never passes it) is byte-identical to before this parameter was added.
    Generalized (Stage H3B) so scripts/run_layer_attention_feature_pilot.py
    can reuse this exact prompt-construction logic for any of the 6
    Stage-H tasks instead of reimplementing it."""
    import pred_long_bench as plb
    from utils.generation_semantics import NO_BUILD_CHAT_DATASETS

    with open(os.path.join(REPO_ROOT, "config", "dataset2prompt.json"), "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    prompt = dataset2prompt[task].format(**json_obj)
    if task not in NO_BUILD_CHAT_DATASETS:
        prompt = plb.build_chat(tokenizer, prompt, model_short_name)
    return prompt


def load_canary_model(
    model_name_or_path, cache_dir, k_bits, v_bits, seed,
    layer_idx=CANARY_LAYER, group_size=CANARY_GROUP_SIZE, residual_length=CANARY_RESIDUAL_LENGTH,
    family="kivi",
):
    """`layer_idx`/`group_size`/`residual_length`/`family` default to the
    exact H2 canary constants (family="kivi") so every existing H2/H3 call
    site (which never passes them) is byte-identical to before these
    parameters were added. Generalized (Stage H3B: layer_idx/group_size/
    residual_length; Stage I1-C: family) so the scientific collector and
    the Rotation-KIVI canary can load any of the 8 primary layers, under
    any SUPPORTED_FAMILIES value, through this exact same, already-
    validated model-construction path -- never an independent
    reimplementation. All non-target layers are ALWAYS family="kivi"
    K16/V16, regardless of what the target layer's family/bits are."""
    import torch
    import pred_long_bench as plb

    plb.seed_everything(seed)
    probe_config = plb.LlamaConfig.from_pretrained(model_name_or_path)
    policy_obj = {
        "policy_name": "h0_attention_decode_canary",
        "default": {"k_bits": 16, "v_bits": 16, "family": "kivi"},
        "overrides": {str(layer_idx): {"k_bits": k_bits, "v_bits": v_bits, "family": family}},
    }
    resolved = resolve_layer_policy(probe_config.num_hidden_layers, 16, 16, policy_obj)

    model_args = _Args()
    model_args.model_name_or_path = model_name_or_path
    model_args.k_bits = 16
    model_args.v_bits = 16
    model_args.group_size = group_size
    model_args.residual_length = residual_length
    training_args = _Args()
    training_args.cache_dir = cache_dir
    dtype = torch.float16

    model, tokenizer, model_class_name = plb.build_model_and_tokenizer(
        model_args, training_args, dtype, use_kivi_model=True, resolved_layer_policy=resolved
    )
    model.eval()
    attn = model.model.layers[layer_idx].self_attn
    got = (attn.k_bits, attn.v_bits, attn.family)
    if got != (k_bits, v_bits, family):
        raise CanaryError(f"layer {layer_idx} policy mismatch: expected {(k_bits, v_bits, family)}, got {got}")
    for i, l in enumerate(model.model.layers):
        if i == layer_idx:
            continue
        g = (l.self_attn.k_bits, l.self_attn.v_bits, l.self_attn.family)
        if g != (16, 16, "kivi"):
            raise CanaryError(f"layer {i} is not K16/V16 family='kivi' control: got {g}")
    return model, tokenizer, model_class_name


def generate_with_capture(model, tokenizer, prompt, max_new_tokens, capture=True, layer_idx=CANARY_LAYER, extra_generate_kwargs=None, capture_cls=LayerCallCapture):
    """`layer_idx`/`extra_generate_kwargs`/`capture_cls` default to
    CANARY_LAYER/None/LayerCallCapture so every existing H2 call site
    (which never passes them) issues the exact same model.generate() call,
    using the exact same capture class, as before these parameters were
    added. Generalized (Stage H3B: layer_idx/extra_generate_kwargs; Stage
    I1-C0V2: capture_cls) so the scientific collector can target any layer
    and pass task-specific generation kwargs (e.g. samsum's min_length/
    eos_token_id, via utils.generation_semantics), and so a narrowly-scoped
    LayerCallCapture SUBCLASS (never a reimplementation) can additionally
    observe production-internal calls (e.g. apply_hadamard_rotation) --
    through this same fresh-hook-state-per-call path, rather than
    reimplementing any of it."""
    import torch

    extra_generate_kwargs = extra_generate_kwargs or {}
    attn = model.model.layers[layer_idx].self_attn
    layer_capture = capture_cls(attn, attn.head_dim, attn.num_key_value_heads) if capture else None
    if capture:
        layer_capture.install_handles()

    inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inp, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False, temperature=1.0, top_p=1.0,
            **extra_generate_kwargs,
        )
    generated_ids = output[0, inp.input_ids.shape[1]:].tolist()

    if capture:
        layer_capture.remove_handles()
    return generated_ids, layer_capture, inp.input_ids.shape[1]


def independent_q_reconstruction(call, num_heads, head_dim):
    """Second, genuinely independent path to the post-RoPE decode query:
    starts from the raw q_proj hook (captured separately from the RoPE
    spy) and reapplies RoPE itself by calling the real, unmodified
    _apply_rotary_pos_emb_inplace directly (not via the scoped spy) with
    the same captured cos/sin/position_ids/chunk_size. Compared against
    the RoPE-spy-captured Q_t in the parity report -- proves Q parity via
    two different capture mechanisms, not by re-observing one spy twice.
    """
    import models.llama_kivi as llama_kivi

    import torch

    raw_q = call["q_proj_raw"]
    bsz, q_len, _ = raw_q.shape
    raw_q = raw_q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2).contiguous()
    dummy_k = raw_q.clone()
    # Force the same no-grad chunked in-place branch production actually
    # used at capture time (model.generate() runs under no_grad) -- avoids
    # relying on the grad-enabled/no-grad branch-equivalence proof from
    # Stage G1 to justify this comparison; it exercises the identical path.
    with torch.no_grad():
        q_out, _ = llama_kivi._apply_rotary_pos_emb_inplace(
            raw_q.clone(), dummy_k, call["cos"], call["sin"], call["position_ids_in"], call["rotary_chunk_size"]
        )
    return q_out


def reconstruct_key_step(call, group_size, k_bits, shadow, head_dim, num_kv_heads):
    import torch
    import models.llama_kivi as llama_kivi

    Q_t = call["q_post_rope"]
    K_t = call["k_post_rope"]
    past = parse_kivi_cache_tuple(call["past_key_value_in"])

    shadow.append_decode_step(K_t, call["v_current"])

    if past.key_quant_trans is not None:
        att_qkquant = llama_kivi.cuda_bmm_fA_qB_outer(group_size, Q_t, past.key_quant_trans, past.key_scale_trans, past.key_mn_trans, k_bits)
    else:
        att_qkquant = None
    key_full_with_current = torch.cat([past.key_full, K_t], dim=2) if past.key_full is not None else K_t
    att_qkfull = torch.matmul(Q_t, llama_kivi.repeat_kv(key_full_with_current, 1).transpose(2, 3))
    if att_qkquant is not None:
        L_kivi = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(head_dim)
    else:
        L_kivi = att_qkfull / math.sqrt(head_dim)

    L_fp16 = torch.matmul(Q_t, shadow.k().transpose(2, 3)) / math.sqrt(head_dim)
    return L_kivi, L_fp16


def reconstruct_value_step(call, group_size, v_bits, shadow, head_dim):
    import torch
    import models.llama_kivi as llama_kivi

    Q_t = call["q_post_rope"]
    K_shadow_full = shadow.k()  # includes current step (appended by reconstruct_key_step-equivalent logic below)
    past = parse_kivi_cache_tuple(call["past_key_value_in"])

    A_fp16 = torch.nn.functional.softmax((torch.matmul(Q_t, K_shadow_full.transpose(2, 3)) / math.sqrt(head_dim)).float(), dim=-1).to(Q_t.dtype)
    O_fp16 = torch.matmul(A_fp16, shadow.v())

    value_full_with_current = torch.cat([past.value_full, call["v_current"]], dim=2) if past.value_full is not None else call["v_current"]
    value_full_length = value_full_with_current.shape[-2]
    if past.value_quant is not None:
        O_value_kivi = llama_kivi.cuda_bmm_fA_qB_outer(
            group_size, A_fp16[:, :, :, :-value_full_length], past.value_quant, past.value_scale, past.value_mn, v_bits
        )
        O_value_kivi = O_value_kivi + torch.matmul(A_fp16[:, :, :, -value_full_length:], llama_kivi.repeat_kv(value_full_with_current, 1))
    else:
        O_value_kivi = torch.matmul(A_fp16, value_full_with_current)
    return O_value_kivi, O_fp16, A_fp16


def run_axis_canary(axis, model_name_or_path, cache_dir):
    import torch

    k_bits, v_bits = (2, 16) if axis == "key" else (16, 2)
    policy_label = f"K{k_bits}/V{v_bits}"

    model, tokenizer, model_class_name = load_canary_model(model_name_or_path, cache_dir, k_bits, v_bits, CANARY_SEED)
    from datasets import load_dataset

    data = load_dataset("THUDM/LongBench", CANARY_TASK, split="test", trust_remote_code=True)
    json_obj = data[CANARY_DATASET_INDEX]
    model_short_name = model_name_or_path.split("/")[-1]
    prompt = build_prompt(tokenizer, model_short_name, json_obj)

    # --- Hook-neutrality: run once WITH capture, once WITHOUT, compare tokens ---
    ids_with_hooks, layer_capture, prompt_len = generate_with_capture(model, tokenizer, prompt, CANARY_MAX_NEW_TOKENS, capture=True)
    ids_without_hooks, _, _ = generate_with_capture(model, tokenizer, prompt, CANARY_MAX_NEW_TOKENS, capture=False)
    hook_neutrality = {
        "ids_with_hooks": ids_with_hooks,
        "ids_without_hooks": ids_without_hooks,
        "identical": ids_with_hooks == ids_without_hooks,
    }

    attn = model.model.layers[CANARY_LAYER].self_attn
    head_dim = attn.head_dim
    num_kv_heads = attn.num_key_value_heads

    calls = layer_capture.calls
    if len(calls) < 3:
        raise CanaryError(f"expected at least 3 layer-0 forward calls (prefill + >=2 decode steps), got {len(calls)}")

    # attach reshaped V to each call, in call order
    for i, call in enumerate(calls):
        raw_v = call["v_proj_raw"]
        q_len = raw_v.shape[1]
        call["v_current"] = raw_v.view(raw_v.shape[0], q_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    prefill_call = calls[0]
    decode_calls = calls[1:]  # decode step t == decode_calls[t-1]

    shadow = ShadowKVCache()
    shadow.seed_prefill(prefill_call["k_post_rope"], prefill_call["v_current"])

    q_parity_records = []
    key_logit_parity_records = []
    value_output_parity_records = []
    key_step_distortions = {}
    value_step_distortions = {}
    rollover_observations = []

    for step_idx, call in enumerate(decode_calls, start=1):
        spy_q = call["q_post_rope"]  # captured via the transparent RoPE spy
        independent_q = independent_q_reconstruction(call, attn.num_heads, head_dim)
        q_parity_records.append(
            {"step": step_idx, "check": "spy_q_vs_independent_q_proj_plus_rope", **tensor_diff_report(independent_q, spy_q, f"Q_step{step_idx}")}
        )

        if axis == "key":
            L_kivi, L_fp16 = reconstruct_key_step(call, CANARY_GROUP_SIZE, k_bits, shadow, head_dim, num_kv_heads)
            real_logits = call.get("pre_softmax_logits")
            if real_logits is not None:
                key_logit_parity_records.append({"step": step_idx, **tensor_diff_report(L_kivi, real_logits, f"key_logits_step{step_idx}")})
            key_step_distortions[step_idx] = key_decode_distortion(L_kivi, L_fp16)
            past = parse_kivi_cache_tuple(call["past_key_value_in"])
            rollover_observations.append(
                {
                    "step": step_idx,
                    "key_quant_trans_present": past.key_quant_trans is not None,
                    "key_full_len_pre_step": 0 if past.key_full is None else int(past.key_full.shape[-2]),
                }
            )
        else:
            shadow.append_decode_step(call["k_post_rope"], call["v_current"])
            O_value_kivi, O_fp16, A_fp16_manual = reconstruct_value_step(call, CANARY_GROUP_SIZE, v_bits, shadow, head_dim)
            real_output = call.get("attn_output_pre_o_proj")
            if real_output is not None:
                bsz, nh, qlen, hd = O_value_kivi.shape
                o_value_kivi_flat = O_value_kivi.transpose(1, 2).contiguous().view(bsz, qlen, nh * hd)
                value_output_parity_records.append(
                    {"step": step_idx, **tensor_diff_report(o_value_kivi_flat, real_output, f"value_output_step{step_idx}")}
                )
            real_softmax_out = call.get("post_softmax_weights")
            if real_softmax_out is not None:
                q_parity_records.append(
                    {"step": step_idx, "check": "A_fp16_manual_vs_real_softmax", **tensor_diff_report(A_fp16_manual, real_softmax_out, f"A_fp16_step{step_idx}")}
                )
            value_step_distortions[step_idx] = value_decode_distortion(O_value_kivi, O_fp16)
            past = parse_kivi_cache_tuple(call["past_key_value_in"])
            rollover_observations.append(
                {
                    "step": step_idx,
                    "value_quant_present": past.value_quant is not None,
                    "value_full_len_pre_step": 0 if past.value_full is None else int(past.value_full.shape[-2]),
                }
            )

    aggregation = aggregate_sample_decode_distortion(key_step_distortions if axis == "key" else value_step_distortions)

    # --- Generation-horizon parity: truncated vs full lcc-semantics prefix ---
    from utils.generation_semantics import resolve_generate_kwargs

    with open(os.path.join(REPO_ROOT, "config", "dataset2maxlen.json"), "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)
    max_gen_full = dataset2maxlen[CANARY_TASK]
    inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    context_length = inp.input_ids.shape[-1]
    full_kwargs = resolve_generate_kwargs(CANARY_TASK, tokenizer, context_length, max_gen_full)
    with torch.no_grad():
        full_output = model.generate(**inp, **full_kwargs)
    full_ids = full_output[0, context_length:].tolist()
    common_len = min(len(full_ids), len(ids_with_hooks))
    horizon_prefix_match = full_ids[:common_len] == ids_with_hooks[:common_len]

    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "axis": axis,
        "policy": policy_label,
        "model_class": model_class_name,
        "num_layer0_calls": len(calls),
        "num_decode_steps_captured": len(decode_calls),
        "hook_neutrality": hook_neutrality,
        "shadow_token_count_final": shadow.token_count,
        "q_parity_records": q_parity_records,
        "key_logit_parity_records": key_logit_parity_records,
        "value_output_parity_records": value_output_parity_records,
        "rollover_observations": rollover_observations,
        "aggregation": aggregation,
        "prompt_len": prompt_len,
        "generated_ids_truncated": ids_with_hooks,
        "generated_ids_full_task": full_ids,
        "generated_ids_full_task_len": len(full_ids),
        "horizon_common_len": common_len,
        "horizon_prefix_match": horizon_prefix_match,
    }


def main():
    import torch

    if not torch.cuda.is_available():
        raise CanaryError("CUDA is not available; this canary requires a real GPU.")

    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = os.path.join(DEFAULT_OUTPUT_ROOT, run_label)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "host_monitor.log")

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    boot_id_start = get_boot_id()
    gpu_before = gpu_snapshot()
    log(f"start run_label={run_label} boot_id={boot_id_start}")

    model_name_or_path = "lmsys/longchat-7b-v1.5-32k"
    cache_dir = "./cached_models"

    summary = {
        "run_label": run_label,
        "canary_task": CANARY_TASK,
        "canary_dataset_index": CANARY_DATASET_INDEX,
        "canary_layer": CANARY_LAYER,
        "boot_id_start": boot_id_start,
        "git_commit": get_git_commit(),
        "git_status_short": get_git_status_short(),
        "cuda_device": torch.cuda.get_device_name(0),
        "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
        "gpu_before": gpu_before,
    }

    exit_code = 0
    try:
        log("running Key-axis (K2/V16) canary")
        summary["key_axis"] = run_axis_canary("key", model_name_or_path, cache_dir)
        log("running Value-axis (K16/V2) canary")
        summary["value_axis"] = run_axis_canary("value", model_name_or_path, cache_dir)
    except Exception as e:  # noqa: BLE001 -- canary must record, not crash uncaught
        summary["error"] = f"{type(e).__name__}: {e}"
        exit_code = 1

    boot_id_end = get_boot_id()
    gpu_after = gpu_snapshot()
    summary["boot_id_end"] = boot_id_end
    summary["boot_id_stable"] = boot_id_start == boot_id_end
    summary["gpu_after"] = gpu_after
    summary["exit_code"] = exit_code

    with open(os.path.join(out_dir, "parity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"end exit_code={exit_code}")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nArtifacts written to: {out_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
