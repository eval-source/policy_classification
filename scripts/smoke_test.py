"""Smoke test: confirm Tinker sampling + chat template works on Qwen/Qwen3.5-4B."""
import _env  # noqa: F401  (loads API key)
import tinker
from tinker import types

MODEL = "Qwen/Qwen3.5-4B"


def main():
    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(base_model=MODEL)
    tok = sc.get_tokenizer()

    messages = [{"role": "user", "content": "Reply with exactly one word: OK"}]
    # Non-thinking mode for a fast, direct answer (hybrid model supports this).
    try:
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=True
        )
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)

    prompt = types.ModelInput.from_ints(ids)
    params = types.SamplingParams(max_tokens=16, temperature=0.0)
    resp = sc.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
    out = tok.decode(resp.sequences[0].tokens)
    print("STOP REASON:", resp.sequences[0].stop_reason)
    print("OUTPUT:", repr(out))
    print("OK: sampling works")


if __name__ == "__main__":
    main()
