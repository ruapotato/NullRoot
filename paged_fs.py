"""
Paged filesystem: each directory is its own context string.

The model only sees the current directory's page. cd swaps pages.
The page table (dict of path → context) is managed by Python.
All operations are local to the current page.

Page format:
    children:foo bar baz#foo>content of foo#bar>content of bar

    - children:NAME NAME NAME — list of entries (dirs end with /, files don't)
    - NAME>CONTENT — file content (only for files in this directory)
    - Variables: $key=value (global, prepended to every page)
"""

from tokenizer import BashTokenizer


class PagedFileSystem:
    """Paged filesystem where each directory is a separate context string."""

    def __init__(self):
        # Page table: absolute path → page content
        # Page content is the serialized directory state
        self.pages: dict[str, dict] = {
            "/": {"children": [], "files": {}},
        }
        self.cwd = "/"
        self.vars: dict[str, str] = {}
        self.exit_status = 0

    def _ensure_page(self, path: str):
        """Create a page if it doesn't exist."""
        if path not in self.pages:
            self.pages[path] = {"children": [], "files": {}}

    def _current_page(self) -> dict:
        return self.pages[self.cwd]

    def _child_path(self, name: str) -> str:
        if self.cwd == "/":
            return f"/{name}"
        return f"{self.cwd}/{name}"

    # --- Serialize current page to context string ---

    def serialize_page(self) -> str:
        """Serialize current directory page + variables."""
        page = self._current_page()
        parts = []

        # Variables (global)
        parts.append(f"$?={self.exit_status}")
        for k in sorted(self.vars):
            parts.append(f"${k}={self.vars[k]}")

        # Children list (dirs marked with /)
        child_names = []
        for name in sorted(page["children"]):
            child_path = self._child_path(name)
            if child_path in self.pages:
                child_names.append(f"{name}/")
            else:
                child_names.append(name)
        parts.append(f"children:{' '.join(child_names)}")

        # File contents
        for name in sorted(page["files"]):
            content = page["files"][name].replace("\\", "\\\\").replace("\n", "\\n")
            parts.append(f"{name}>{content}")

        return "#".join(parts)

    def serialize_full_input(self, cmd: str) -> str:
        """Build full model input: @CWD#page_state + prompt."""
        page_state = self.serialize_page()
        return f"<state>@{self.cwd}#{page_state}<eor><prompt>{cmd}<eoi>"

    # --- Parse patch and apply ---

    def apply_patch(self, patch: str):
        """Apply a model-generated patch to the current page."""
        if not patch:
            return

        page = self._current_page()

        for part in patch.split("#"):
            if not part:
                continue

            if part.startswith("$?="):
                self.exit_status = int(part[3:]) if part[3:].isdigit() else 0
            elif part.startswith("$"):
                eq = part.index("=") if "=" in part else len(part)
                key = part[1:eq]
                val = part[eq + 1:] if eq < len(part) else ""
                self.vars[key] = val
            elif part.startswith("children:"):
                names = part[9:].split() if part[9:] else []
                page["children"] = [n.rstrip("/") for n in names]
                # Create pages for new directories
                for n in names:
                    if n.endswith("/"):
                        dir_name = n.rstrip("/")
                        child_path = self._child_path(dir_name)
                        self._ensure_page(child_path)
            elif part.startswith("-"):
                # Deletion
                name = part[1:]
                if name in page["files"]:
                    del page["files"][name]
                if name in page["children"]:
                    page["children"].remove(name)
            elif ">" in part:
                name = part[:part.index(">")]
                content = part[part.index(">") + 1:]
                content = content.replace("\\n", "\n").replace("\\\\", "\\")
                page["files"][name] = content
                if name not in page["children"]:
                    page["children"].append(name)

    # --- Commands (reference implementation) ---

    def execute(self, cmd: str) -> tuple[str, str]:
        """Execute command, return (response, patch).

        Response is the text output. Patch is the state change string.
        """
        cmd = self._expand_vars(cmd)
        parts = cmd.split()
        if not parts:
            return "", ""

        op = parts[0]
        page = self._current_page()

        if op == "ls":
            entries = sorted(page["children"])
            display = []
            for name in entries:
                child_path = self._child_path(name)
                if child_path in self.pages:
                    display.append(name)
                else:
                    display.append(name)
            return "  ".join(display), ""

        elif op == "pwd":
            return self.cwd, ""

        elif op == "mkdir" and len(parts) > 1:
            name = parts[1]
            if name in page["children"]:
                return "", ""  # already exists
            page["children"].append(name)
            child_path = self._child_path(name)
            self._ensure_page(child_path)
            # Patch: updated children list
            return "", self._children_patch()

        elif op == "touch" and len(parts) > 1:
            name = parts[1]
            if name not in page["files"]:
                page["files"][name] = ""
                if name not in page["children"]:
                    page["children"].append(name)
            return "", self._children_patch()

        elif op == "echo" and (">" in parts or ">>" in parts):
            if ">>" in parts:
                idx = parts.index(">>")
                content = " ".join(parts[1:idx])
                name = parts[idx + 1]
                if name in page["files"]:
                    page["files"][name] += "\n" + content
                else:
                    page["files"][name] = content
                    if name not in page["children"]:
                        page["children"].append(name)
            else:
                idx = parts.index(">")
                content = " ".join(parts[1:idx])
                name = parts[idx + 1]
                page["files"][name] = content
                if name not in page["children"]:
                    page["children"].append(name)
            escaped = page["files"][name].replace("\\", "\\\\").replace("\n", "\\n")
            return "", f"{self._children_patch()}#{name}>{escaped}"

        elif op == "cat" and len(parts) > 1:
            name = parts[1]
            if name in page["files"]:
                return page["files"][name], ""
            return "", ""

        elif op == "rm" and len(parts) > 1:
            name = parts[1]
            if name in page["files"]:
                del page["files"][name]
            if name in page["children"]:
                page["children"].remove(name)
                # Also remove page if it was a directory
                child_path = self._child_path(name)
                self.pages.pop(child_path, None)
            return "", f"{self._children_patch()}#-{name}"

        elif op == "cp" and len(parts) > 2:
            src, dst = parts[1], parts[2]
            if src in page["files"]:
                page["files"][dst] = page["files"][src]
                if dst not in page["children"]:
                    page["children"].append(dst)
                escaped = page["files"][dst].replace("\\", "\\\\").replace("\n", "\\n")
                return "", f"{self._children_patch()}#{dst}>{escaped}"
            return "", ""

        elif op == "mv" and len(parts) > 2:
            src, dst = parts[1], parts[2]
            if src in page["files"]:
                page["files"][dst] = page["files"][src]
                del page["files"][src]
                if src in page["children"]:
                    page["children"].remove(src)
                if dst not in page["children"]:
                    page["children"].append(dst)
                escaped = page["files"][dst].replace("\\", "\\\\").replace("\n", "\\n")
                return "", f"{self._children_patch()}#{dst}>{escaped}#-{src}"
            return "", ""

        elif op == "head" and len(parts) > 1:
            name = parts[1]
            if name in page["files"]:
                lines = page["files"][name].split("\n")
                return lines[0], ""
            return "", ""

        elif op == "wc" and len(parts) > 1:
            name = parts[1]
            if name in page["files"]:
                content = page["files"][name]
                lines = content.count("\n") + (1 if content else 0)
                words = len(content.split())
                chars = len(content)
                return f"{lines} {words} {chars} {name}", ""
            return "", ""

        elif op == "grep" and len(parts) > 2:
            pattern, name = parts[1], parts[2]
            if name in page["files"]:
                matches = [l for l in page["files"][name].split("\n") if pattern in l]
                return "\n".join(matches), ""
            return "", ""

        elif op == "cd" and len(parts) > 1:
            target = parts[1]
            if target == "..":
                if self.cwd == "/":
                    return "", ""
                self.cwd = "/".join(self.cwd.rstrip("/").split("/")[:-1]) or "/"
            elif target.startswith("/"):
                if target in self.pages:
                    self.cwd = target
            else:
                child_path = self._child_path(target)
                if child_path in self.pages:
                    self.cwd = child_path
            # cd outputs the new page as a full state replacement
            return "", self.serialize_page()

        elif op == "expr" and len(parts) > 1:
            expr_str = " ".join(parts[1:])
            try:
                tokens = expr_str.split()
                if len(tokens) == 3:
                    a, sym, b = int(tokens[0]), tokens[1], int(tokens[2])
                    if sym == "+": return str(a + b), ""
                    if sym == "-": return str(a - b), ""
                    if sym == "*": return str(a * b), ""
            except (ValueError, IndexError):
                pass
            return "", ""

        elif "=" in op and not op.startswith("="):
            name, _, value = op.partition("=")
            self.vars[name] = self._expand_vars(value)
            return "", f"${name}={self.vars[name]}"

        elif op == "export" and len(parts) > 1 and "=" in parts[1]:
            name, _, value = parts[1].partition("=")
            self.vars[name] = value
            return "", f"${name}={value}"

        elif op == "echo":
            return " ".join(parts[1:]), ""

        return "", ""

    def _children_patch(self) -> str:
        """Generate children: patch for current page."""
        page = self._current_page()
        child_names = []
        for name in sorted(page["children"]):
            child_path = self._child_path(name)
            if child_path in self.pages:
                child_names.append(f"{name}/")
            else:
                child_names.append(name)
        return f"children:{' '.join(child_names)}"

    def _expand_vars(self, text: str) -> str:
        result = []
        i = 0
        while i < len(text):
            if text[i] == "$":
                j = i + 1
                while j < len(text) and (text[j].isalnum() or text[j] in "_?"):
                    j += 1
                var_name = text[i + 1:j]
                if var_name == "?":
                    result.append(str(self.exit_status))
                elif var_name:
                    result.append(self.vars.get(var_name, ""))
                else:
                    result.append("$")
                i = j
            else:
                result.append(text[i])
                i += 1
        return "".join(result)


if __name__ == "__main__":
    print("Testing PagedFileSystem...")
    fs = PagedFileSystem()

    tests = [
        ("mkdir etc", "", "has children"),
        ("mkdir home", "", "has children"),
        ("ls", "etc  home", "lists both"),
        ("cd etc", "", "page swap"),
        ("ls", "", "empty dir"),
        ("echo 127.0.0.1 localhost > hosts", "", "create file"),
        ("ls", "hosts", "shows file"),
        ("cat hosts", "127.0.0.1 localhost", "reads content"),
        ("pwd", "/etc", "correct path"),
        ("cd ..", "", "go up"),
        ("pwd", "/", "back at root"),
        ("ls", "etc  home", "root still has both"),
        ("cd home", "", "enter home"),
        ("mkdir alice", "", "create subdir"),
        ("cd alice", "", "enter alice"),
        ("echo hello world > readme.txt", "", "create file"),
        ("cat readme.txt", "hello world", "read it back"),
        ("pwd", "/home/alice", "deep path"),
        ("x=42", "", "set var"),
        ("echo $x", "42", "expand var"),
        ("expr 5 + 3", "8", "math"),
    ]

    passed = 0
    for cmd, expected, desc in tests:
        response, patch = fs.execute(cmd)
        if patch:
            fs.apply_patch(patch)
        ok = response == expected
        passed += ok
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {cmd:40s} -> {response!r:30s} ({desc})")
        if not ok:
            print(f"         expected: {expected!r}")

    print(f"\n  {passed}/{len(tests)} passed")
    print(f"\n  Pages: {list(fs.pages.keys())}")
    print(f"  Current page ({fs.cwd}): {fs.serialize_page()}")
