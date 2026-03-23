"""
Interactive inference for BashTransformer (Phase 2 — memory-based).

Each command is processed in isolation. Memory carries filesystem state
across commands within the session. No token history — only the current
command tokens and memory are fed to the model.
"""

import argparse
import sys

import torch
from torch.amp import autocast

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig


def load_model(ckpt_path: str, device: torch.device) -> BashTransformer:
    """Load a trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", BashTransformerConfig())
    model = BashTransformer(config).to(device).to(torch.bfloat16)
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    step = ckpt.get("global_step", ckpt.get("step", "?"))
    print(f"Loaded checkpoint from step {step}")
    return model


@torch.no_grad()
def generate(
    model: BashTransformer,
    cmd_ids: list[int],
    memory: torch.Tensor,
    tokenizer: BashTokenizer,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[list[int], torch.Tensor]:
    """Generate tokens for a single command using memory.

    Returns (generated_ids, updated_memory).
    """
    eor_id = tokenizer.eor_id
    eos_id = tokenizer.eos_id

    ids = list(cmd_ids)
    generated = []

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, memory=memory)
        logits = out["logits"][0, -1, :]
        memory = out["memory"]

        if temperature <= 0:
            next_id = logits.argmax().item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)
        generated.append(next_id)

        if next_id == eor_id or next_id == eos_id:
            break

    return generated, memory


def interactive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BashTokenizer()
    model = load_model(args.checkpoint, device)

    # Fresh memory for the session
    memory = model.reset_memory(1, device, next(model.parameters()).dtype)

    print(f"\nBashTransformer interactive mode (Phase 2 — memory-based)")
    print(f"Memory: DeltaNet state {model.config.state_heads}h x {model.config.state_head_dim}d")
    print(f"Type bash commands. 'reset' to clear memory. Ctrl-C or 'exit' to quit.\n")

    while True:
        try:
            cmd = input("$ ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd.strip() == "exit":
            break

        if cmd.strip() == "reset":
            memory = model.reset_memory(1, device, next(model.parameters()).dtype)
            print("(memory reset)")
            continue

        # Encode: <prompt>command<eoi>
        cmd_text = f"<prompt>{cmd}<eoi>"
        cmd_ids = tokenizer.encode(cmd_text)

        # Generate response using only this command + memory
        generated, memory = generate(
            model, cmd_ids, memory, tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )

        # After generation, update memory with the full exchange
        # (command + generated response) so future commands see it
        full_ids = cmd_ids + generated
        input_t = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(input_t, memory=memory)
        memory = out["memory"]

        # Decode and display
        response_text = tokenizer.decode(generated)
        display = response_text
        for tag in ["<output>", "<err>", "<eor>", "<prompt>", "<eos>", "<eoi>"]:
            display = display.replace(tag, "")
        display = display.strip()

        if display:
            print(display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive BashTransformer inference (Phase 2)")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens to generate per response")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    args = parser.parse_args()
    interactive(args)
