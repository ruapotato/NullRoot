#!/usr/bin/env python3
"""
NullRoot Simulator — reference implementation of the NullRoot shell.

This is the "perfect" version that the neural network is learning to emulate.
Same state format, same commands, deterministic execution. Use this to
explore what the system can do and verify behavior.

Usage:
    python nullroot_sim.py
    python nullroot_sim.py --demo unix
    python nullroot_sim.py --demo project
"""

import argparse
from generator import FileSystem


DEMOS = {
    "unix": [
        "mkdir etc",
        "mkdir home",
        "mkdir tmp",
        "mkdir var",
        "cd etc",
        "echo 127.0.0.1 localhost > hosts",
        "echo root 0 0 > passwd",
        "echo nameserver 8.8.8.8 > resolv.conf",
        "mkdir ssh",
        "cd ssh",
        "echo port 22 > sshd_config",
        "cd /home",
        "mkdir alice",
        "mkdir bob",
        "cd alice",
        "mkdir documents",
        "mkdir projects",
        "echo hello world > readme.txt",
        "cd documents",
        "echo meeting notes from monday > notes.txt",
        "echo todo fix the build > todo.txt",
        "echo todo update docs >> todo.txt",
        "cd /home/bob",
        "mkdir scripts",
        "cd scripts",
        "echo echo backup started > backup.sh",
        "echo echo deploy done > deploy.sh",
        "cd /var",
        "mkdir log",
        "cd log",
        "echo system started > syslog",
        "echo login alice >> syslog",
        "echo login bob >> syslog",
        "cd /",
    ],
    "project": [
        "mkdir src",
        "mkdir tests",
        "mkdir docs",
        "mkdir config",
        "echo print hello > src/main.py",
        "echo import main > src/init.py",
        "echo assert true > tests/test.py",
        "echo name myapp version 1 > config/app.cfg",
        "echo build dist .env > .gitignore",
        "echo my project readme > readme.md",
        "touch changelog.md",
        "cd /",
    ],
    "scripting": [
        "echo echo hello world > greet.sh",
        "echo echo the time is now > status.sh",
        'echo "echo files\\nls\\necho done" > report.sh',
        "mkdir data",
        "cd data",
        "echo alice 42 > users.txt",
        "echo bob 37 >> users.txt",
        "echo charlie 25 >> users.txt",
        "cd /",
        "x=10",
        "name=nullroot",
        "version=3",
    ],
}


def run_demo(fs, demo_name):
    if demo_name not in DEMOS:
        print(f"Unknown demo: {demo_name}")
        print(f"Available: {', '.join(DEMOS.keys())}")
        return

    commands = DEMOS[demo_name]
    print(f"Building '{demo_name}' filesystem ({len(commands)} commands)...")
    for cmd in commands:
        output = fs.execute_command(cmd)
        # Don't print output during setup

    print(f"Ready. {len(fs.dirs)} dirs, {len(fs.files)} files, {len(fs.vars)} vars.")
    print(f"CWD: {fs.cwd}\n")


def main():
    parser = argparse.ArgumentParser(description="NullRoot Simulator")
    parser.add_argument("--demo", "-d", type=str, default=None,
                        choices=list(DEMOS.keys()),
                        help="Pre-build a filesystem")
    args = parser.parse_args()

    fs = FileSystem()

    if args.demo:
        run_demo(fs, args.demo)

    print("NullRoot Simulator (reference implementation)")
    print("Commands: mkdir cd ls pwd touch echo cat rm cp mv head wc find grep")
    print("          expr <math>, x=val, echo $x, sh script.sh, export name=val")
    print("Type 'state' for raw state, 'reset' to clear, 'exit' to quit.\n")

    while True:
        try:
            cmd = input(f"nullroot-sim:{fs.cwd}$ ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        cmd = cmd.strip()
        if not cmd:
            continue
        if cmd == "exit":
            break
        if cmd == "reset":
            fs = FileSystem()
            print("(reset)")
            continue
        if cmd == "state":
            print(f"  {fs.serialize_state()}")
            continue

        output = fs.execute_command(cmd)
        if output:
            print(output)


if __name__ == "__main__":
    main()
