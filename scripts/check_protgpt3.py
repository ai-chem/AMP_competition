"""Smoke-check ProtGPT3-1.3B tokenization and generation.

Usage:
    uv run python scripts/check_protgpt3.py
    uv run python scripts/check_protgpt3.py --tokenizer-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import (
    DEFAULT_SEED,
    REPO_ROOT,
    finish_run,
    load_config,
    save_run,
    seed_everything,
)
from amp_competition.constants import STANDARD_AMINO_ACIDS
from amp_competition.generator.protgpt3 import (
    MODEL_AMINO_ACIDS,
    decode_protein,
    encode_prompt,
    generate_sequences,
    is_challenge_alphabet,
    is_model_alphabet,
    load_model,
    load_tokenizer,
)

EXPECTED_VOCAB = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "[UNK]": 3,
    "1": 4,
    "2": 5,
    "A": 6,
    "B": 7,
    "C": 8,
    "D": 9,
    "E": 10,
    "F": 11,
    "G": 12,
    "H": 13,
    "I": 14,
    "K": 15,
    "L": 16,
    "M": 17,
    "N": 18,
    "O": 19,
    "P": 20,
    "Q": 21,
    "R": 22,
    "S": 23,
    "T": 24,
    "U": 25,
    "V": 26,
    "W": 27,
    "X": 28,
    "Y": 29,
    "Z": 30,
}

PREFIX = "1MKT"
EXPECTED_PREFIX_IDS = [1, 4, 17, 15, 24]  # bos, 1, M, K, T


def _ok(passed: bool, label: str, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{mark}] {label}{suffix}")
    return passed


def check_tokenizer(tokenizer) -> bool:
    print("\n=== Tokenization ===")
    ok = True

    vocab = tokenizer.get_vocab()
    ok &= _ok(len(vocab) >= 31, "vocab covers 31 model tokens", f"len={len(vocab)}")
    mismatches = [token for token, idx in EXPECTED_VOCAB.items() if vocab.get(token) != idx]
    ok &= _ok(not mismatches, "expected token ids", f"mismatches={mismatches[:8]}")

    bos_on_empty = tokenizer("", add_special_tokens=True).input_ids
    ok &= _ok(
        bos_on_empty[:1] == [tokenizer.bos_token_id],
        "empty prompt starts with BOS",
        f"ids={bos_on_empty}",
    )

    prefix_ids = tokenizer(PREFIX, add_special_tokens=True).input_ids
    ok &= _ok(
        prefix_ids == EXPECTED_PREFIX_IDS,
        f"encode({PREFIX!r}) with BOS",
        f"got={prefix_ids} expected={EXPECTED_PREFIX_IDS}",
    )

    explicit = encode_prompt(tokenizer, PREFIX).tolist()[0]
    ok &= _ok(explicit == EXPECTED_PREFIX_IDS, "encode_prompt prepends BOS", f"ids={explicit}")

    aa_ids = []
    for residue in STANDARD_AMINO_ACIDS:
        ids = tokenizer(residue, add_special_tokens=False).input_ids
        aa_ids.append((residue, ids))
    ok &= _ok(
        all(len(ids) == 1 for _, ids in aa_ids),
        "each standard amino acid is one token",
        f"bad={[res for res, ids in aa_ids if len(ids) != 1]}",
    )

    direction_ids = {
        "1": tokenizer("1", add_special_tokens=False).input_ids,
        "2": tokenizer("2", add_special_tokens=False).input_ids,
    }
    ok &= _ok(
        direction_ids == {"1": [4], "2": [5]},
        "direction tokens 1=N→C and 2=C→N",
        f"ids={direction_ids}",
    )

    decoded = tokenizer.decode(EXPECTED_PREFIX_IDS, skip_special_tokens=True)
    ok &= _ok(decoded.replace(" ", "") == PREFIX, "roundtrip decode prefix", f"got={decoded!r}")

    decoded_protein = decode_protein(decoded)
    ok &= _ok(
        decoded_protein.direction == "N2C" and decoded_protein.sequence == "MKT",
        "decode_protein strips direction 1",
        f"{decoded_protein}",
    )
    reversed_c2n = decode_protein("2TKM")
    ok &= _ok(
        reversed_c2n.direction == "C2N" and reversed_c2n.sequence == "MKT",
        "decode_protein reverses C→N",
        f"{reversed_c2n}",
    )

    unk_ids = tokenizer("j*", add_special_tokens=False).input_ids
    ok &= _ok(
        all(idx == tokenizer.unk_token_id or tokenizer.convert_ids_to_tokens(idx) == "[UNK]" for idx in unk_ids),
        "out-of-vocab characters map to UNK",
        f"ids={unk_ids} tokens={tokenizer.convert_ids_to_tokens(unk_ids)}",
    )
    return ok


def check_generation(tokenizer, model, generation_cfg: dict) -> bool:
    print("\n=== Generation ===")
    import torch

    device = next(model.parameters()).device
    print(f"device={device} dtype={next(model.parameters()).dtype}")
    ok = True

    n_seq = 4
    texts = generate_sequences(
        tokenizer,
        model,
        PREFIX,
        num_return_sequences=n_seq,
        max_new_tokens=int(generation_cfg.get("max_length", 50)),
        temperature=float(generation_cfg.get("temperature", 0.8)),
        top_p=float(generation_cfg.get("top_p", 0.9)),
        seed=42,
    )
    ok &= _ok(len(texts) == n_seq, f"returned {n_seq} sequences", f"n={len(texts)}")

    prefix_ok = 0
    alphabet_ok = 0
    challenge_ok = 0
    continued = 0
    for i, text in enumerate(texts, start=1):
        parsed = decode_protein(text)
        compact = parsed.raw
        has_prefix = compact.startswith(PREFIX)
        in_alphabet = is_model_alphabet(parsed.sequence)
        in_challenge = is_challenge_alphabet(parsed.sequence)
        grew = len(parsed.sequence) > len("MKT")
        prefix_ok += int(has_prefix)
        alphabet_ok += int(in_alphabet)
        challenge_ok += int(in_challenge)
        continued += int(grew)
        print(
            f"  gen{i}: dir={parsed.direction} len={len(parsed.sequence)} "
            f"prefix={has_prefix} model_aa={in_alphabet} challenge_aa={in_challenge} "
            f"text={compact[:80]!r}"
        )

    ok &= _ok(prefix_ok == n_seq, "prefix 1MKT preserved", f"{prefix_ok}/{n_seq}")
    ok &= _ok(alphabet_ok == n_seq, "continuation in ProtGPT3 alphabet", f"{alphabet_ok}/{n_seq}")
    ok &= _ok(continued >= 1, "at least one sequence continued past the prefix", f"{continued}/{n_seq}")
    _ok(
        challenge_ok == n_seq,
        "all residues in AMP Challenge 20-AA alphabet (informational)",
        f"{challenge_ok}/{n_seq}",
    )

    blank = generate_sequences(
        tokenizer,
        model,
        "",
        num_return_sequences=2,
        max_new_tokens=24,
        temperature=0.8,
        top_p=0.9,
        seed=0,
    )
    directed = 0
    for i, text in enumerate(blank, start=1):
        parsed = decode_protein(text)
        directed += int(parsed.direction in {"N2C", "C2N"})
        print(
            f"  uncond{i}: dir={parsed.direction} len={len(parsed.sequence)} "
            f"text={parsed.raw[:80]!r}"
        )
    ok &= _ok(directed == len(blank), "unconditional samples start with 1 or 2", f"{directed}/{len(blank)}")

    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"cuda max allocated: {allocated:.2f} GiB")
    return ok


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="check_protgpt3.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Check ProtGPT3 tokenization and generation.")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--seed", type=int, default=int(config.get("seed", DEFAULT_SEED)))
    args = parser.parse_args()

    resolved = dict(config)
    resolved["seed"] = args.seed
    out_dir = REPO_ROOT / "outputs" / "check_protgpt3"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=bool(resolved.get("deterministic", True)))
    run = save_run("check_protgpt3", resolved, out_dir=out_dir)

    model_cfg = resolved.get("model", {})
    model_id = model_cfg.get("id", "AI4PD/ProtGPT3-1.3B")
    print(f"model={model_id} seed={args.seed} run_id={run['run_id']}")

    status = "failed"
    try:
        tokenizer = load_tokenizer(
            model_id,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
        tok_ok = check_tokenizer(tokenizer)
        if args.tokenizer_only:
            status = "completed" if tok_ok else "failed"
            finish_run(run, extra={"tokenizer_ok": tok_ok, "tokenizer_only": True}, status=status)
            print("\nTOKENIZER:", "OK" if tok_ok else "FAILED")
            return 0 if tok_ok else 1

        model = load_model(
            model_id,
            dtype=str(model_cfg.get("dtype", "bfloat16")),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
        gen_ok = check_generation(tokenizer, model, resolved.get("generation", {}))
        all_ok = tok_ok and gen_ok
        status = "completed" if all_ok else "failed"
        finish_run(run, extra={"tokenizer_ok": tok_ok, "generation_ok": gen_ok}, status=status)
        print("\nOVERALL:", "OK" if all_ok else "FAILED")
        return 0 if all_ok else 1
    except Exception:
        finish_run(run, status=status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
