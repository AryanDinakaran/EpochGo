import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional


class CodeExecutor:
    """Safely executes generated Python scripts inside the project directory."""

    @staticmethod
    def execute_script(
        script_path: Path,
        cwd: Optional[Path] = None,
        timeout_seconds: int = 300,
    ) -> Dict[str, Any]:
        if not script_path.exists():
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Script not found: {script_path}",
                "exit_code": -1,
                "duration": 0,
            }

        script_path = script_path.resolve()
        working_dir = (cwd or script_path.parent).resolve()
        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            duration = round(time.time() - start_time, 2)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "duration": duration,
            }
        except subprocess.TimeoutExpired:
            duration = round(time.time() - start_time, 2)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds.",
                "exit_code": -1,
                "duration": duration,
            }
        except Exception as e:
            duration = round(time.time() - start_time, 2)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution failed with exception: {str(e)}",
                "exit_code": -1,
                "duration": duration,
            }
