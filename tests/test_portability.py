"""验证工程离开主仓库仍可导入；不运行收费服务或 Qlib 训练。"""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_no_python_module_imports_original_rdagent(self):
        violations = []
        for path in (PROJECT / "factor_agent").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(m == "rdagent" or m.startswith("rdagent.") for m in modules):
                    violations.append(f"{path.relative_to(PROJECT)}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_copy_imports_all_modules_with_rdagent_forbidden(self):
        """Import hook also catches dynamic imports and installed-package fallback."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(PROJECT / "factor_agent", root / "factor_agent",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            code = """
import importlib
import importlib.abc
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
class BlockOriginal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rdagent' or fullname.startswith('rdagent.'):
            raise AssertionError('original framework dependency: ' + fullname)
sys.meta_path.insert(0, BlockOriginal())
import factor_agent
assert Path(factor_agent.__file__).resolve().is_relative_to(Path.cwd())
for path in sorted(Path('factor_agent').glob('*.py')):
    if path.stem not in ('__main__',):
        importlib.import_module('factor_agent.' + path.stem)
print('standalone imports passed')
"""
            completed = subprocess.run([sys.executable, "-I", "-c", code], cwd=root,
                                       text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("standalone imports passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
