"""Thin wrapper around the `colab` CLI (google-colab-cli)."""

import subprocess


class ColabSessionError(Exception):
    """A `colab` CLI invocation failed."""


class ColabSession:
    """Runs colab-cli commands for one named session."""

    def __init__(self, sessionName: str, colabBin: str = "colab") -> None:
        self.sessionName = sessionName
        self.colabBin = colabBin

    def _run(
        self,
        args: list[str],
        stdin: str | None = None,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self.colabBin, *args]
        if capture:
            return subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                check=False,
            )
        # Stream the CLI's output live so the user sees it working.
        return subprocess.run(
            cmd,
            input=stdin,
            text=True,
            check=False,
        )

    def _checked(
        self,
        args: list[str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = self._run(args, stdin=stdin)
        if result.returncode != 0:
            raise ColabSessionError(
                f"`{' '.join([self.colabBin, *args])}` failed "
                f"with exit code {result.returncode}",
            )
        return result

    def new(
        self,
        gpu: str = "G4",
        highMem: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        args = ["new", "-s", self.sessionName, "--gpu", gpu]
        if highMem:
            args.append("--high-mem")
        return self._checked(args)

    def stop(self) -> subprocess.CompletedProcess[str]:
        return self._checked(["stop", "-s", self.sessionName])

    def status(self) -> subprocess.CompletedProcess[str]:
        return self._checked(["status", "-s", self.sessionName])

    def install(self, packages: list[str]) -> subprocess.CompletedProcess[str]:
        return self._checked(["install", "-s", self.sessionName, *packages])

    def execCode(
        self,
        code: str,
        timeout: float | None = None,
        capture: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        args = ["exec", "-s", self.sessionName]
        if timeout is not None:
            args += ["--timeout", str(timeout)]
        if capture:
            return self._run(args, stdin=code, capture=True)
        return self._checked(args, stdin=code)

    def upload(
        self,
        localPath: str,
        remotePath: str,
    ) -> subprocess.CompletedProcess[str]:
        return self._checked(
            ["upload", "-s", self.sessionName, localPath, remotePath],
        )
