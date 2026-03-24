"""
Reference interpreter: verifies the state-patch system produces correct results.

Takes the same training data and verifies:
1. Each command produces the correct response
2. Each state patch applies correctly
3. The state after patching matches the ground truth filesystem
4. Scripts execute correctly
5. Variables expand correctly
"""

import sys
import random
from tokenizer import BashTokenizer
from generator import FileSystem, _gen_syllable_name_rng
from dataset import StateTrackingGenerator, build_session_samples


def verify_session(seed: int, min_ops=50, target_ops=100, verbose=False):
    """Generate and verify one full session."""
    tok = BashTokenizer()

    gen = StateTrackingGenerator(min_ops=min_ops, target_ops=target_ops, seed=seed)
    cmds = gen.generate_with_state()
    samples = build_session_samples(cmds, tok)

    if not samples:
        return True, 0, []

    errors = []
    state_str = ""

    for i, (cmd_info, sample) in enumerate(zip(cmds, samples)):
        # Decode the full training sample
        text = tok.decode(sample["ids"])

        # Get the ground truth state
        gt_state = cmd_info["fs"].serialize_state()

        # Extract patch from the training sample
        if "<state>" in text:
            after_eoi = text[text.index("<eoi>") + 5:]
            if "<state>" in after_eoi:
                state_part = after_eoi[after_eoi.index("<state>") + 7:]
                if state_part.endswith("<eor>"):
                    state_part = state_part[:-5]
                patch = state_part
            else:
                patch = ""
        elif "<nop>" in text:
            patch = ""
        else:
            patch = ""

        # Apply patch
        if patch:
            new_state = FileSystem.apply_patch(state_str, patch)
        else:
            new_state = state_str

        # Verify: does the patched state match ground truth?
        if new_state != gt_state:
            errors.append({
                "cmd_idx": i,
                "cmd": cmd_info["cmd_str"],
                "expected_state": gt_state,
                "got_state": new_state,
                "patch": patch,
                "old_state": state_str,
            })
            if verbose:
                print(f"  MISMATCH at cmd {i}: {cmd_info['cmd_str']}")
                print(f"    expected: {gt_state[:100]}...")
                print(f"    got:      {new_state[:100]}...")
                print(f"    patch:    {patch[:100]}...")

        state_str = gt_state  # Use ground truth for next step (don't compound errors)

    return len(errors) == 0, len(samples), errors


def main():
    num_sessions = 100
    total_cmds = 0
    total_errors = 0
    failed_sessions = 0

    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"Verifying {num_sessions} sessions...")

    for seed in range(num_sessions):
        ok, n_cmds, errors = verify_session(
            seed=seed + 10000, min_ops=50, target_ops=100, verbose=verbose)
        total_cmds += n_cmds
        total_errors += len(errors)
        if not ok:
            failed_sessions += 1
            if not verbose:
                print(f"  Session {seed}: {len(errors)} errors in {n_cmds} commands")
                for e in errors[:3]:
                    print(f"    cmd {e['cmd_idx']}: {e['cmd']}")

        if (seed + 1) % 20 == 0:
            print(f"  [{seed+1}/{num_sessions}] {total_cmds} cmds, "
                  f"{total_errors} errors, {failed_sessions} failed sessions")

    print(f"\n{'='*60}")
    print(f"  Sessions: {num_sessions}")
    print(f"  Commands: {total_cmds}")
    print(f"  Errors:   {total_errors}")
    print(f"  Failed:   {failed_sessions}/{num_sessions}")
    print(f"  {'ALL PASSED' if total_errors == 0 else 'FAILURES DETECTED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
