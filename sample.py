"""
Interactive inference for BashTransformer.

Loads a checkpoint, optionally seeds context from a session transcript,
then accepts user commands and generates responses.

NOTE: During inference, <err> exchanges are stripped from context before
each forward pass to preserve context length. (Not yet implemented —
placeholder for future optimization.)
"""

import argparse
import sys

import torch

from tokenizer import BashTokenizer
from model import BashTransformer, BashTransformerConfig


def load_model(ckpt_path: str, device: torch.device) -> BashTransformer:
    """Load a trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", BashTransformerConfig())
    model = BashTransformer(config).to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint from step {step}")
    return model


@torch.no_grad()
def generate(
    model: BashTransformer,
    token_ids: list[int],
    tokenizer: BashTokenizer,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> list[int]:
    """Generate tokens until <prompt>, <eos>, or max length.

    Stops when the model emits a <prompt> token (it's the user's turn)
    or <eos>.
    """
    prompt_id = tokenizer.prompt_id
    eos_id = tokenizer.eos_id

    generated = []

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        out = model(input_ids)
        logits = out["logits"][0, -1, :]  # last position

        if temperature <= 0:
            next_id = logits.argmax().item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        token_ids.append(next_id)
        generated.append(next_id)

        # Stop on <prompt> (user's turn) or <eos>
        if next_id == prompt_id or next_id == eos_id:
            break

    return generated


def interactive(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BashTokenizer()
    model = load_model(args.checkpoint, device)

    # Seed context from a transcript file if provided
    context_ids: list[int] = []
    if args.context:
        with open(args.context) as f:
            import json
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    record = json.loads(line)
                    text = record.get("transcript", line)
                else:
                    text = line
                context_ids = tokenizer.encode(text)
                break
        # Strip <eos> from end of context so we can continue
        if context_ids and context_ids[-1] == tokenizer.eos_id:
            context_ids = context_ids[:-1]
        print(f"Loaded context: {len(context_ids)} tokens")

    print(f"\nBashTransformer interactive mode")
    print(f"Type bash commands. Ctrl-C or 'exit' to quit.\n")

    while True:
        try:
            cmd = input("$ ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd.strip() == "exit":
            break

        # Encode: <prompt>command\n
        cmd_text = f"<prompt>{cmd}\n"
        cmd_ids = tokenizer.encode(cmd_text)
        context_ids.extend(cmd_ids)

        # Generate response
        generated = generate(
            model, context_ids, tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )

        # Decode and display the response (skip the <output>/<err> token itself)
        response_text = tokenizer.decode(generated)

        # Clean up for display
        display = response_text
        for tag in ["<output>", "<err>", "<prompt>", "<eos>"]:
            display = display.replace(tag, "")
        display = display.strip()

        if display:
            print(display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive BashTransformer inference")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--context", "-c", type=str, default=None,
                        help="JSONL file with a session transcript for pre-context")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens to generate per response")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    args = parser.parse_args()
    interactive(args)
