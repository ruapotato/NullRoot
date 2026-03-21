"""
Correctness verifier for TTY transcripts.

Parses raw TTY transcripts (with <prompt>, <output>, <err>, <eos> tokens),
replays every command against an independent FileSystem simulation, and
checks that every output matches ground truth.
"""

import re
import json
import os
import random
import sys

from generator import FileSystem, SessionGenerator


# ---------------------------------------------------------------------------
# 1. Parser: raw transcript -> list of (command, output_or_none, is_error)
# ---------------------------------------------------------------------------

def parse_transcript(transcript: str) -> list[tuple[str, str | None, bool]]:
    """Parse a TTY transcript into a list of (command, output_or_none, is_error) tuples.

    Format:
        <prompt>command
        <output>content     (normal output with content)
        <output>            (normal output, no content — silent command)
        <err>               (error output)
    Ends with <eos>.
    """
    # Strip trailing <eos> and whitespace
    text = transcript.strip()
    if text.endswith("<eos>"):
        text = text[:-5].strip()

    exchanges = []

    # Split on <prompt> to get each exchange
    # The first split element before the first <prompt> is empty (or whitespace)
    parts = text.split("<prompt>")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Each part is: "command\n<output>content" or "command\n<err>" or "command\n<output>"
        if "\n<err>" in part:
            # Error case: command\n<err>\n... (possibly trailing)
            idx = part.index("\n<err>")
            command = part[:idx].strip()
            exchanges.append((command, None, True))
        elif "\n<output>" in part:
            idx = part.index("\n<output>")
            command = part[:idx].strip()
            rest = part[idx + len("\n<output>"):]
            # rest may be empty (silent command) or contain output content
            # Output content follows immediately after <output>, terminated by
            # a single trailing newline (the delimiter).  We must NOT strip
            # leading newlines — they can be part of the content.
            if rest == "" or rest == "\n":
                output = None
            else:
                # Remove exactly the trailing delimiter newline
                if rest.endswith("\n"):
                    output = rest[:-1]
                else:
                    output = rest
                if output == "":
                    output = None
            exchanges.append((command, output, False))
        elif "<output>" in part:
            # Handle case where <output> is on same line as command (no newline between)
            idx = part.index("<output>")
            command = part[:idx].strip()
            rest = part[idx + len("<output>"):]
            if rest == "" or rest == "\n":
                output = None
            else:
                if rest.endswith("\n"):
                    output = rest[:-1]
                else:
                    output = rest
                if output == "":
                    output = None
            exchanges.append((command, output, False))
        elif "<err>" in part:
            idx = part.index("<err>")
            command = part[:idx].strip()
            exchanges.append((command, None, True))
        else:
            # Bare command with no response marker — shouldn't happen in valid transcript
            command = part.strip()
            if command:
                exchanges.append((command, None, False))

    return exchanges


# ---------------------------------------------------------------------------
# 2. Command parser helpers
# ---------------------------------------------------------------------------

_ECHO_WRITE_RE = re.compile(r'^echo\s+(.+?)\s+>\s+(\S+)$')
_ECHO_APPEND_RE = re.compile(r'^echo\s+(.+?)\s+>>\s+(\S+)$')


def parse_command(cmd: str) -> tuple[str, list[str]]:
    """Parse a command string into (verb, args).

    Returns:
        (verb, args) where verb is one of: ls, pwd, cat, mkdir, touch,
        echo_write, echo_append, cd, rm, unknown
    """
    cmd = cmd.strip()

    # echo ... >> file
    m = _ECHO_APPEND_RE.match(cmd)
    if m:
        return "echo_append", [m.group(1), m.group(2)]

    # echo ... > file
    m = _ECHO_WRITE_RE.match(cmd)
    if m:
        return "echo_write", [m.group(1), m.group(2)]

    parts = cmd.split()
    if not parts:
        return "unknown", []

    verb = parts[0]

    if verb == "ls":
        return "ls", parts[1:]
    elif verb == "pwd":
        return "pwd", []
    elif verb == "cat":
        return "cat", parts[1:]
    elif verb == "mkdir":
        return "mkdir", parts[1:]
    elif verb == "touch":
        return "touch", parts[1:]
    elif verb == "cd":
        return "cd", parts[1:]
    elif verb == "rm":
        return "rm", parts[1:]
    else:
        return "unknown", parts


# ---------------------------------------------------------------------------
# 3. Replayer / Verifier
# ---------------------------------------------------------------------------

class Mismatch:
    """Records a single mismatch between transcript and simulation."""

    def __init__(self, index: int, command: str, field: str, expected, actual):
        self.index = index
        self.command = command
        self.field = field
        self.expected = expected
        self.actual = actual

    def __str__(self):
        return (
            f"  [{self.index}] {self.command!r}\n"
            f"    {self.field}:\n"
            f"      expected: {self.expected!r}\n"
            f"      actual:   {self.actual!r}"
        )


def verify_transcript(transcript: str) -> tuple[int, list[Mismatch]]:
    """Replay a transcript against a fresh FileSystem and verify every exchange.

    Returns:
        (total_commands, list_of_mismatches)
    """
    exchanges = parse_transcript(transcript)
    fs = FileSystem()
    mismatches = []

    for i, (cmd_str, transcript_output, transcript_is_error) in enumerate(exchanges):
        verb, args = parse_command(cmd_str)

        if verb == "ls":
            # ls (no args = cwd)
            if transcript_is_error:
                # ls shouldn't normally error in our simulation
                mismatches.append(Mismatch(i, cmd_str, "unexpected error", "no error for ls", "got <err>"))
                continue

            entries = fs.list_dir(fs.cwd)
            expected_output = "  ".join(entries) if entries else None

            if expected_output != transcript_output:
                mismatches.append(Mismatch(i, cmd_str, "output", expected_output, transcript_output))

        elif verb == "pwd":
            if transcript_is_error:
                mismatches.append(Mismatch(i, cmd_str, "unexpected error", "no error for pwd", "got <err>"))
                continue

            expected_output = fs.cwd
            if expected_output != transcript_output:
                mismatches.append(Mismatch(i, cmd_str, "output", expected_output, transcript_output))

        elif verb == "cat":
            if not args:
                mismatches.append(Mismatch(i, cmd_str, "parse", "cat requires filename", "no args"))
                continue

            filename = args[0]
            content, err = fs.cat(filename)

            if transcript_is_error:
                # Transcript says error — verify that cat should indeed fail
                if err is None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        "cat should succeed (file exists)",
                        "transcript says <err>"
                    ))
            else:
                # Transcript says success
                if err is not None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        f"cat should fail: {err}",
                        "transcript says success"
                    ))
                else:
                    # Compare content
                    # Empty file: content is "", transcript_output is None
                    sim_output = content if content else None
                    if sim_output != transcript_output:
                        mismatches.append(Mismatch(i, cmd_str, "output", sim_output, transcript_output))

        elif verb == "mkdir":
            if not args:
                mismatches.append(Mismatch(i, cmd_str, "parse", "mkdir requires dirname", "no args"))
                continue

            dirname = args[0]
            err = fs.mkdir(dirname)

            if transcript_is_error:
                if err is None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        "mkdir should succeed",
                        "transcript says <err>"
                    ))
            else:
                if err is not None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        f"mkdir should fail: {err}",
                        "transcript says success"
                    ))

        elif verb == "touch":
            if not args:
                mismatches.append(Mismatch(i, cmd_str, "parse", "touch requires filename", "no args"))
                continue

            filename = args[0]
            err = fs.touch(filename)

            if transcript_is_error:
                if err is None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        "touch should succeed",
                        "transcript says <err>"
                    ))
            else:
                if err is not None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        f"touch should fail: {err}",
                        "transcript says success"
                    ))

        elif verb == "echo_write":
            content_str, filename = args[0], args[1]
            err = fs.write_file(filename, content_str)

            if transcript_is_error:
                if err is None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        "echo > should succeed",
                        "transcript says <err>"
                    ))
            else:
                if err is not None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        f"echo > should fail: {err}",
                        "transcript says success"
                    ))
                else:
                    # Verify the file now contains the content
                    abspath = fs.resolve(filename)
                    actual_content = fs.files.get(abspath)
                    if actual_content != content_str:
                        mismatches.append(Mismatch(
                            i, cmd_str, "file content after write",
                            content_str, actual_content
                        ))

        elif verb == "echo_append":
            content_str, filename = args[0], args[1]
            # Capture state before append for verification
            abspath = fs.resolve(filename)
            old_content = fs.files.get(abspath)

            err = fs.append_file(filename, content_str)

            if transcript_is_error:
                if err is None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        "echo >> should succeed",
                        "transcript says <err>"
                    ))
            else:
                if err is not None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        f"echo >> should fail: {err}",
                        "transcript says success"
                    ))
                else:
                    # Verify the file content was correctly appended
                    actual_content = fs.files.get(abspath)
                    if old_content is not None:
                        expected_content = old_content + "\n" + content_str
                    else:
                        expected_content = content_str
                    if actual_content != expected_content:
                        mismatches.append(Mismatch(
                            i, cmd_str, "file content after append",
                            expected_content, actual_content
                        ))

        elif verb == "cd":
            if not args:
                mismatches.append(Mismatch(i, cmd_str, "parse", "cd requires path", "no args"))
                continue

            path = args[0]
            old_cwd = fs.cwd

            if transcript_is_error:
                # The transcript says error. Verify cd should indeed fail.
                # Special case: cd .. from root is treated as error by the generator
                if path == ".." and fs.cwd == "/":
                    # This is the "cd_up_from_root" invalid op — don't change fs state
                    pass
                else:
                    # Try the cd to see if it would fail
                    abspath = fs.resolve(path)
                    if fs.exists_dir(abspath):
                        mismatches.append(Mismatch(
                            i, cmd_str, "error mismatch",
                            f"cd should succeed (dir {abspath} exists)",
                            "transcript says <err>"
                        ))
                    # Don't actually cd since it was an error
            else:
                # Transcript says success
                err = fs.cd(path)
                if err is not None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        f"cd should fail: {err}",
                        "transcript says success"
                    ))

        elif verb == "rm":
            if not args:
                mismatches.append(Mismatch(i, cmd_str, "parse", "rm requires filename", "no args"))
                continue

            filename = args[0]

            if transcript_is_error:
                # Verify rm should indeed fail
                abspath = fs.resolve(filename)
                if abspath in fs.files:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        "rm should succeed (file exists)",
                        "transcript says <err>"
                    ))
                # Don't actually rm since it was an error
            else:
                err = fs.rm(filename)
                if err is not None:
                    mismatches.append(Mismatch(
                        i, cmd_str, "error mismatch",
                        f"rm should fail: {err}",
                        "transcript says success"
                    ))
                else:
                    # Verify file is actually gone
                    abspath = fs.resolve(filename)
                    if abspath in fs.files:
                        mismatches.append(Mismatch(
                            i, cmd_str, "rm verification",
                            "file should be removed",
                            "file still exists"
                        ))

        elif verb == "unknown":
            mismatches.append(Mismatch(i, cmd_str, "unknown command", "recognized command", cmd_str))

    return len(exchanges), mismatches


# ---------------------------------------------------------------------------
# 4. Reporting
# ---------------------------------------------------------------------------

def print_report(label: str, total: int, mismatches: list[Mismatch]):
    """Print a detailed verification report."""
    print(f"\n{'=' * 70}")
    print(f"  VERIFICATION REPORT: {label}")
    print(f"{'=' * 70}")
    print(f"  Total commands replayed: {total}")
    print(f"  Mismatches found:       {len(mismatches)}")

    if mismatches:
        print(f"\n  MISMATCHES:")
        print(f"  {'-' * 66}")
        for m in mismatches:
            print(m)
        print(f"\n  RESULT: FAIL")
    else:
        print(f"\n  RESULT: PASS")

    print(f"{'=' * 70}")
    return len(mismatches) == 0


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_passed = True

    # --- Test 1: Generate a fresh session with 500+ ops and verify it ---
    print("Generating a session with 500+ ops...")
    random.seed(12345)
    gen = SessionGenerator(min_ops=500, target_ops=600, error_rate=0.05)
    transcript = gen.generate()

    num_prompts = transcript.count("<prompt>")
    print(f"  Generated {num_prompts} commands.")

    total, mismatches = verify_transcript(transcript)
    passed = print_report("Generated Session (500+ ops)", total, mismatches)
    if not passed:
        all_passed = False

    # --- Test 2: Load and verify data/validation.jsonl if it exists ---
    validation_path = os.path.join(os.path.dirname(__file__) or ".", "data", "validation.jsonl")
    if os.path.exists(validation_path):
        print(f"\nLoading {validation_path}...")
        with open(validation_path, "r") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                t = record["transcript"]
                num_prompts = t.count("<prompt>")
                print(f"  Record {line_num}: {num_prompts} commands")

                total, mismatches = verify_transcript(t)
                label = f"validation.jsonl record {line_num}"
                passed = print_report(label, total, mismatches)
                if not passed:
                    all_passed = False
    else:
        print(f"\n{validation_path} not found — skipping.")

    # --- Test 3: Additional edge-case micro-tests ---
    print("\nRunning edge-case micro-tests...")

    edge_cases_passed = 0
    edge_cases_total = 0

    def micro_test(label: str, transcript_str: str, expect_pass: bool = True):
        global edge_cases_passed, edge_cases_total
        edge_cases_total += 1
        total, mm = verify_transcript(transcript_str)
        ok = (len(mm) == 0) == expect_pass
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {label} ({total} cmds, {len(mm)} mismatches)")
        if not ok:
            for m in mm:
                print(m)
        if ok:
            edge_cases_passed += 1
        return ok

    # Empty directory ls
    micro_test("ls on empty root",
        "<prompt>ls\n<output>\n<eos>")

    # mkdir + ls
    micro_test("mkdir then ls",
        "<prompt>mkdir foo\n<output>\n"
        "<prompt>ls\n<output>foo\n<eos>")

    # touch + cat empty file
    micro_test("touch then cat empty",
        "<prompt>mkdir dir\n<output>\n"
        "<prompt>cd dir\n<output>\n"
        "<prompt>touch f\n<output>\n"
        "<prompt>cat f\n<output>\n<eos>")

    # echo write + cat
    micro_test("echo write then cat",
        "<prompt>echo hello world > greet.txt\n<output>\n"
        "<prompt>cat greet.txt\n<output>hello world\n<eos>")

    # echo append + cat
    micro_test("echo append",
        "<prompt>echo line1 > f.txt\n<output>\n"
        "<prompt>echo line2 >> f.txt\n<output>\n"
        "<prompt>cat f.txt\n<output>line1\nline2\n<eos>")

    # cd + pwd
    micro_test("cd and pwd",
        "<prompt>mkdir a\n<output>\n"
        "<prompt>cd a\n<output>\n"
        "<prompt>pwd\n<output>/a\n<eos>")

    # cd .. back to root
    micro_test("cd .. to root",
        "<prompt>mkdir x\n<output>\n"
        "<prompt>cd x\n<output>\n"
        "<prompt>cd ..\n<output>\n"
        "<prompt>pwd\n<output>/\n<eos>")

    # rm
    micro_test("rm file",
        "<prompt>echo data > tmp.txt\n<output>\n"
        "<prompt>rm tmp.txt\n<output>\n"
        "<prompt>ls\n<output>\n<eos>")

    # Error: cd nonexistent
    micro_test("cd nonexistent error",
        "<prompt>cd nope\n<err>\n<eos>")

    # Error: cat nonexistent
    micro_test("cat nonexistent error",
        "<prompt>cat nope.txt\n<err>\n<eos>")

    # Error: mkdir existing
    micro_test("mkdir existing error",
        "<prompt>mkdir foo\n<output>\n"
        "<prompt>mkdir foo\n<err>\n<eos>")

    # Error: rm nonexistent
    micro_test("rm nonexistent error",
        "<prompt>rm nope.txt\n<err>\n<eos>")

    # Intentional mismatch: wrong ls output
    micro_test("intentional mismatch in ls (expect fail)",
        "<prompt>mkdir aaa\n<output>\n"
        "<prompt>ls\n<output>bbb\n<eos>",
        expect_pass=False)

    # Intentional mismatch: wrong cat output
    micro_test("intentional mismatch in cat (expect fail)",
        "<prompt>echo real content > f.txt\n<output>\n"
        "<prompt>cat f.txt\n<output>wrong content\n<eos>",
        expect_pass=False)

    # Absolute cd
    micro_test("absolute cd",
        "<prompt>mkdir a\n<output>\n"
        "<prompt>cd a\n<output>\n"
        "<prompt>mkdir b\n<output>\n"
        "<prompt>cd /a/b\n<output>\n"
        "<prompt>pwd\n<output>/a/b\n<eos>")

    print(f"\n  Edge-case micro-tests: {edge_cases_passed}/{edge_cases_total} passed")
    if edge_cases_passed != edge_cases_total:
        all_passed = False

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    if all_passed:
        print("  OVERALL: ALL TESTS PASSED")
    else:
        print("  OVERALL: SOME TESTS FAILED")
    print(f"{'=' * 70}")

    sys.exit(0 if all_passed else 1)
