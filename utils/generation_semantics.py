"""Single source of truth for dataset-specific LongBench generation-call
semantics, extracted from pred_long_bench.py's get_pred() so that
scripts/layer_policy_smoke.py, scripts/run_layer_sensitivity_pilot.py, and
any future non-formal generation harness reproduce production generation
behavior exactly instead of independently re-deriving (and risking drift
from) it.

pred_long_bench.py itself is intentionally left unmodified: it already
implements this logic correctly inline, and the validated formal runner
should not be touched when not necessary. This module exists so nothing
else has to duplicate that logic from scratch.
"""

# Mirrors pred_long_bench.py's get_pred() build_chat skip-list exactly:
# "chat models are better off without build prompts on these tasks".
NO_BUILD_CHAT_DATASETS = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]


def resolve_generate_kwargs(dataset, tokenizer, context_length, max_gen):
    """Returns the exact kwargs pred_long_bench.py's get_pred() passes to
    model.generate() for `dataset`, given this sample's tokenized prompt
    length (context_length) and the task's max_gen (from
    config/dataset2maxlen.json).

    samsum gets a min_length + custom eos_token_id guard against the known
    "model endlessly repeats \\nDialogue" failure mode (see
    pred_long_bench.py's inline comment on this exact branch); every other
    dataset gets the generic kwargs only.
    """
    kwargs = {
        "max_new_tokens": max_gen,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if dataset == "samsum":
        kwargs["min_length"] = context_length + 1
        kwargs["eos_token_id"] = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
    return kwargs
