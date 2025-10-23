import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

from epm.input_diagnostic import run_diagnostics


class InputDiagnosticSmokeTest(unittest.TestCase):
    def test_default_data_test_folder(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        gms_path = repo_root / "epm" / "input_readers.gms"
        args = SimpleNamespace(
            input_readers=str(gms_path),
            root_input="input",
            folder="data_test",
            show_loaded=False,
        )

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = run_diagnostics(args)

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Input diagnostics for input/data_test", output)
        self.assertIn("pDemandData", output)


if __name__ == "__main__":
    unittest.main()

