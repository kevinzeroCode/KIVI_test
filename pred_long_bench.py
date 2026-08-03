import gc
import os
import subprocess
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
os.environ["WANDB_DISABLED"] = "true"

import transformers
from utils.process_args import process_args
from transformers import LlamaConfig, MistralConfig, AutoTokenizer


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    # Update: we add the template for the new llama-3-instruct model
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def int_env(name, default):
    return int(os.environ.get(name, default))


def configure_kivi_memory_chunks(config):
    config.key_quant_chunk_size = int_env("KIVI_KEY_QUANT_CHUNK", 512)
    config.value_quant_chunk_size = int_env("KIVI_VALUE_QUANT_CHUNK", 512)
    config.rotary_chunk_size = int_env("KIVI_ROTARY_CHUNK", 2048)
    config.mlp_chunk_size = int_env("KIVI_MLP_CHUNK", 2048)
    config.norm_chunk_size = int_env("KIVI_NORM_CHUNK", config.mlp_chunk_size)


# Bit widths the KIVI CUDA/Triton kernels actually support (2, 4) plus 16,
# which means "keep this side full precision, never quantize it" (see
# quantize_key/quantize_value in models/llama_kivi.py). Any other value must
# fail here, before a model is loaded, rather than falling into the CUDA
# kernel's `else` branch (quant/csrc/gemv_cuda.cu), which silently assumes
# 2-bit packing for anything that isn't 4-bit.
ALLOWED_BITS = {2, 4, 16}

# Output roots can be overridden for smoke tests / small functional tests so
# they never write into the real baseline pred/ or pred_e/ directories.
PRED_ROOT = os.environ.get("KIVI_PRED_ROOT", "pred")
PRED_E_ROOT = os.environ.get("KIVI_PRED_E_ROOT", "pred_e")

RUN_CONFIG_CORE_KEYS = [
    "model_name_or_path", "k_bits", "v_bits", "group_size", "residual_length", "max_length",
]


def validate_bits(k_bits, v_bits):
    if k_bits not in ALLOWED_BITS or v_bits not in ALLOWED_BITS:
        raise ValueError(
            f"Unsupported bit configuration k_bits={k_bits}, v_bits={v_bits}. "
            f"Allowed values are {sorted(ALLOWED_BITS)}."
        )


def loader_decision(k_bits, v_bits):
    """Pure decision logic for which model class a (k_bits, v_bits) config uses.

    Only k_bits == v_bits == 16 uses the plain HF model (use_kivi_model=False);
    every other allowed combination -- both sides quantized, or exactly one
    side quantized -- uses LlamaForCausalLM_KIVI, which internally decides per
    side (quantize_key/quantize_value) whether to quantize or pass through.
    """
    quantize_key = k_bits < 16
    quantize_value = v_bits < 16
    use_kivi_model = quantize_key or quantize_value
    return quantize_key, quantize_value, use_kivi_model


def build_pred_dir_name(model_name, max_length, k_bits, v_bits, group_size, residual_length):
    """Output prediction directory name for a given bit config.

    Symmetric configs (k_bits == v_bits) keep the legacy naming scheme so the
    existing FP16/KIVI-2/KIVI-4 baseline directories are unaffected. Mixed
    configs (k_bits != v_bits) use a scheme that encodes both bit widths so
    they can never collide with a symmetric directory (e.g. K2/V16 must not
    land in the same directory as the K2/V2 baseline).
    """
    if k_bits == v_bits:
        return f"{model_name}_{max_length}_{k_bits}bits_group{group_size}_residual{residual_length}"
    return f"{model_name}_{max_length}_k{k_bits}_v{v_bits}_group{group_size}_residual{residual_length}"


def get_git_commit():
    try:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def build_run_config(model_args, max_length, model_class_name, quantize_key, quantize_value, seed):
    return {
        "model_name_or_path": model_args.model_name_or_path,
        "k_bits": model_args.k_bits,
        "v_bits": model_args.v_bits,
        "group_size": model_args.group_size,
        "residual_length": model_args.residual_length,
        "max_length": max_length,
        "seed": seed,
        "model_class": model_class_name,
        "quantize_key": quantize_key,
        "quantize_value": quantize_value,
        "git_commit": get_git_commit(),
        "transformers_version": transformers.__version__,
    }


def load_run_config(pred_dir):
    path = os.path.join(pred_dir, "run_config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_run_config(pred_dir, run_config):
    path = os.path.join(pred_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)


def prepare_run_directory(pred_dir, run_config):
    """Create pred_dir if needed and enforce run_config.json resume safety.

    Returns a status string for logging:
      - "created": brand-new directory, run_config.json just written
      - "validated": existing run_config.json matches the current config
      - "legacy_no_metadata": pre-existing symmetric-bit directory with no
        run_config.json (a baseline completed before this feature existed);
        left untouched and treated as read-only compatible

    Fails closed (raises RuntimeError) if:
      - an existing run_config.json's core settings differ from this run
      - the directory is a mixed-bit (k_bits != v_bits) directory with no
        run_config.json at all, since that can only mean unverified state
    """
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
        write_run_config(pred_dir, run_config)
        return "created"

    existing = load_run_config(pred_dir)
    if existing is not None:
        mismatches = {
            key: (existing.get(key), run_config.get(key))
            for key in RUN_CONFIG_CORE_KEYS
            if existing.get(key) != run_config.get(key)
        }
        if mismatches:
            raise RuntimeError(
                f"Refusing to resume/skip in {pred_dir}: run_config.json does not match "
                f"the current run. Mismatched keys (existing, requested): {mismatches}"
            )
        return "validated"

    if run_config["k_bits"] != run_config["v_bits"]:
        raise RuntimeError(
            f"{pred_dir} is a mixed-bit directory (k_bits={run_config['k_bits']}, "
            f"v_bits={run_config['v_bits']}) with no run_config.json. Refusing to "
            "resume or skip without verified run metadata."
        )
    return "legacy_no_metadata"


def count_jsonl(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path, start_idx=0):
    total = len(data)
    with open(out_path, "a", encoding="utf-8") as f:
        for idx in tqdm(range(start_idx, total), initial=start_idx, total=total):
            json_obj = data[idx]
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            # if "chatglm3" in model:
            #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            with torch.no_grad():
                if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                    output = model.generate(
                        **input,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        min_length=context_length + 1,
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    )[0]
                else:
                    output = model.generate(
                        **input,
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                    )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)
            row = {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}
            json.dump(row, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
            del input, output, tokenized_prompt, row
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # mitigate fragmentation OOM on 24GB GPUs

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(model_args, training_args, dtype, use_kivi_model):
    """Load config/tokenizer/model for the requested (k_bits, v_bits).

    Factored out of __main__ so smoke tests can exercise the exact same
    production loader decision (see loader_decision()) without duplicating
    it or running the full LongBench prediction loop.
    """
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                            use_fast=False,
                                            trust_remote_code=True,
                                            tokenizer_type='llama')
                                            # model_max_length=training_args.model_max_length)
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                            use_fast=False,
                                            trust_remote_code=True)
    else:
        raise NotImplementedError

    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        # Three-way decision: only k_bits==16 and v_bits==16 together use the
        # standard HF model (both sides full precision, no KIVI at all). Any
        # other allowed combination -- both sides quantized, or exactly one
        # side quantized -- goes through LlamaForCausalLM_KIVI, which decides
        # per-side (quantize_key/quantize_value) whether to quantize or keep
        # a full-precision pass-through cache. See models/llama_kivi.py.
        if use_kivi_model:
            from models.llama_kivi import LlamaForCausalLM_KIVI
            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.residual_length = model_args.residual_length
            configure_kivi_memory_chunks(config)
            config.use_flash = True # Note: We activate the flashattention to speed up the inference
            model = LlamaForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            if os.environ.get("KIVI_OFFLOAD_LM_HEAD", "1") != "0":
                model.lm_head = model.lm_head.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )

    elif 'mistral' in model_args.model_name_or_path.lower():
        if model_args.k_bits < 16 and model_args.v_bits < 16:
            from models.mistral_kivi import MistralForCausalLM_KIVI
            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.residual_length = model_args.residual_length
            configure_kivi_memory_chunks(config)
            config.use_flash = True
            model = MistralForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )

    else:
        raise NotImplementedError

    model_class_name = type(model).__name__
    return model, tokenizer, model_class_name


if __name__ == '__main__':
    seed_everything(42)
    # args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    # define your model
    model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    validate_bits(model_args.k_bits, model_args.v_bits)
    quantize_key, quantize_value, use_kivi_model = loader_decision(model_args.k_bits, model_args.v_bits)
    model_name = model_args.model_name_or_path.split("/")[-1]
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    dtype = torch.float16

    model, tokenizer, model_class_name = build_model_and_tokenizer(model_args, training_args, dtype, use_kivi_model)

    #
    # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")

    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    max_length = model2maxlen[model_name]

    def _route_label(quantize_side):
        if quantize_side:
            return "KIVI quantized"
        return "FP16 pass-through" if use_kivi_model else "FP16"

    run_config = build_run_config(model_args, max_length, model_class_name, quantize_key, quantize_value, seed=42)
    pred_dir_name = build_pred_dir_name(
        model_name, max_length, model_args.k_bits, model_args.v_bits, model_args.group_size, model_args.residual_length
    )
    pred_root = PRED_E_ROOT if data_args.e else PRED_ROOT
    if not os.path.exists(pred_root):
        os.makedirs(pred_root)
    pred_dir = os.path.join(pred_root, pred_dir_name)
    resume_status = prepare_run_directory(pred_dir, run_config)

    print(f"Model class: {model_class_name}")
    print(f"K bits: {model_args.k_bits}")
    print(f"V bits: {model_args.v_bits}")
    print(f"quantize_key: {quantize_key}")
    print(f"quantize_value: {quantize_value}")
    print(f"Key route: {_route_label(quantize_key)}")
    print(f"Value route: {_route_label(quantize_value)}")
    print(f"Output directory: {pred_dir}")
    print(f"Resume metadata status: {resume_status}")

    if data_args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "musique", "2wikimqa",
                    "gov_report", "qmsum", "multi_news", "lcc", "repobench-p", "triviaqa",
                    "samsum", "trec", "passage_retrieval_en"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset (pred_dir was already created/validated above)
    for dataset in datasets:
        if data_args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', trust_remote_code=True)
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        out_path = os.path.join(pred_dir, f"{dataset}.jsonl")
        expected = len(data)
        partial_path = f"{out_path}.partial"
        if os.path.exists(out_path):
            done = count_jsonl(out_path)
            if done >= expected:
                print(f"skip {dataset}: {out_path} exists ({done}/{expected})")
                continue
            active_path = out_path
            print(f"resume {dataset}: {done}/{expected} from {out_path}")
        else:
            active_path = partial_path
            done = count_jsonl(partial_path)
            if done:
                print(f"resume {dataset}: {done}/{expected} from {partial_path}")
            else:
                print(f"start {dataset}: writing {partial_path}")
        if done > expected:
            raise ValueError(f"{active_path} has {done} rows, expected at most {expected}")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, active_path, done)
        if active_path != out_path and count_jsonl(active_path) >= expected:
            os.replace(active_path, out_path)
