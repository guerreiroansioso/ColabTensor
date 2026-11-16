"""Orchestrates provisioning the G4 session and serving both models."""

import time
from dataclasses import dataclass, field

from ColabTensor.Backends import BACKENDS, LLAMA_CPP_CUDART_URL, LLAMA_CPP_URL
from ColabTensor.ColabSession import ColabSession, ColabSessionError
from ColabTensor.LocalTunnel import LocalTunnel
from ColabTensor.ModelSpec import DEFAULT_MODELS, MODEL_REGISTRY, ModelSpec

BACKEND = BACKENDS["llama-server"]


@dataclass
class ProvisionResult:
    sessionName: str
    urls: dict[str, str] = field(default_factory=dict)


class Provisioner:
    """Creates the G4 session, serves the models, exposes localtunnel."""

    def __init__(
        self,
        sessionName: str,
        gpu: str = "G4",
        only: str = "both",
        dryRun: bool = False,
        healthTimeoutSec: int = 3600,
    ) -> None:
        self.sessionName = sessionName
        self.gpu = gpu
        self.only = only
        self.dryRun = dryRun
        self.healthTimeoutSec = healthTimeoutSec
        self.session = ColabSession(sessionName)
        self.tunnel = LocalTunnel()

    def _selectSpecs(self) -> list[ModelSpec]:
        names = DEFAULT_MODELS if self.only == "both" else (self.only,)
        return [MODEL_REGISTRY[name] for name in names]

    def _emit(self, message: str) -> None:
        print(f"[colabtensor] {message}")

    def _runOrPrint(self, label: str, action) -> None:
        if self.dryRun:
            self._emit(f"DRY-RUN {label}")
            return
        self._emit(label)
        action()

    def _execChecked(
        self,
        label: str,
        snippet: str,
        marker: str,
        timeout: float | None = None,
    ) -> None:
        if self.dryRun:
            self._emit(f"DRY-RUN {label}")
            return
        self._emit(label)
        out = self.session.execCode(snippet, timeout=timeout, capture=True)
        text = (out.stdout or "") + (out.stderr or "")
        print(text, end="")
        if marker not in text:
            raise ColabSessionError(f"{label} failed; see output above")

    def run(self) -> ProvisionResult:
        specs = self._selectSpecs()
        result = ProvisionResult(sessionName=self.sessionName)
        self._provision()
        self._installPackages()
        self._installLlamaCpp()
        self._downloadModels(specs)
        self._launch(specs)
        self._launchTunnels(specs)
        result.urls = self._collectUrls(specs)
        self._writeUrlsFile(result)
        self._printSummary(specs, result)
        self._keepAlive()
        return result

    def _highMem(self) -> bool:
        # A100 defaults to the 40 GB shape; the models need the 80 GB one.
        return self.gpu.upper() == "A100"

    def _provision(self) -> None:
        highMem = self._highMem()
        suffix = " --high-mem" if highMem else ""
        self._runOrPrint(
            f"colab new -s {self.sessionName} --gpu {self.gpu}{suffix}",
            lambda: self.session.new(self.gpu, highMem=highMem),
        )

    def _installPackages(self) -> None:
        packages = BACKEND.installPackages()
        self._runOrPrint(
            f"colab install -s {self.sessionName} {' '.join(packages)}",
            lambda: self.session.install(packages),
        )

    def _installLlamaCpp(self) -> None:
        snippet = (
            "import urllib.request, tarfile, glob, os, stat\n"
            f'urls = ["{LLAMA_CPP_URL}", "{LLAMA_CPP_CUDART_URL}"]\n'
            "for i, url in enumerate(urls):\n"
            '    dest = f"/content/llama-{i}.tar.gz"\n'
            "    urllib.request.urlretrieve(url, dest)\n"
            "for i in range(len(urls)):\n"
            '    with tarfile.open(f"/content/llama-{i}.tar.gz") as t:\n'
            '        t.extractall("/content/llama", filter="data")\n'
            'matches = glob.glob("/content/llama/**/llama-server", '
            "recursive=True)\n"
            "if not matches:\n"
            '    print("LLAMA_SERVER_NOT_FOUND")\n'
            "    raise SystemExit(1)\n"
            "server = matches[0]\n"
            "os.chmod(server, os.stat(server).st_mode | stat.S_IEXEC)\n"
            'open("/content/llama_server_path.txt", "w").write(server)\n'
            'print("llama-server at", server)\n'
        )
        self._execChecked(
            "download and extract llama.cpp (CUDA 12.8, ~794 MB)",
            snippet,
            "llama-server at",
            timeout=0,
        )

    def _downloadModels(self, specs: list[ModelSpec]) -> None:
        for spec in specs:
            snippet = BACKEND.downloadSnippet(spec)
            self._runOrPrint(
                f"download {spec.hfRepo} ({spec.modelFile})",
                lambda snippet=snippet: self.session.execCode(
                    snippet,
                    timeout=0,
                ),
            )

    def _launch(self, specs: list[ModelSpec]) -> None:
        for spec in specs:
            snippet = BACKEND.launchSnippet(spec)
            self._execChecked(
                f"launch {spec.name} on port {spec.port}",
                snippet,
                "launched",
            )

    def _launchTunnels(self, specs: list[ModelSpec]) -> None:
        self._execChecked(
            "install localtunnel",
            self.tunnel.installSnippet(),
            "localtunnel installed",
            timeout=600,
        )
        for spec in specs:
            snippet = self.tunnel.launchSnippet(spec.port)
            self._execChecked(
                f"localtunnel for port {spec.port}",
                snippet,
                "launched",
            )

    def _healthSnippet(self, spec: ModelSpec) -> str:
        path = BACKEND.healthPath()
        url = f"http://localhost:{spec.port}{path}"
        return (
            "import time\n"
            "import urllib.request\n"
            f"url = '{url}'\n"
            "for _ in range(240):\n"
            "    try:\n"
            "        body = urllib.request.urlopen(url, timeout=5).read()\n"
            "        print(body.decode())\n"
            "        break\n"
            "    except Exception:\n"
            "        print('.', flush=True)\n"
            "        time.sleep(15)\n"
            "else:\n"
            "    print('TIMEOUT')\n"
        )

    def _healthCheckOnceSnippet(self, spec: ModelSpec) -> str:
        path = BACKEND.healthPath()
        url = f"http://localhost:{spec.port}{path}"
        return (
            "import urllib.request\n"
            f"url = '{url}'\n"
            "try:\n"
            "    urllib.request.urlopen(url, timeout=5)\n"
            "    print('HEALTH_OK')\n"
            "except Exception:\n"
            "    print('HEALTH_FAIL')\n"
        )

    def _waitHealthy(self, spec: ModelSpec) -> None:
        if self.dryRun:
            self._emit(f"DRY-RUN wait for {spec.name} health")
            return
        self._emit(f"waiting for {spec.name} to become healthy")
        self.session.execCode(self._healthSnippet(spec), timeout=0)
        out = self.session.execCode(
            self._healthCheckOnceSnippet(spec),
            capture=True,
        )
        if "HEALTH_OK" not in (out.stdout or ""):
            raise ColabSessionError(
                f"{spec.name} did not become healthy; "
                f"see /content/{spec.name}.log on the VM",
            )

    def _pollTunnelUrl(self, port: int, attempts: int = 60) -> str | None:
        for _ in range(attempts):
            out = self.session.execCode(
                self.tunnel.readUrlSnippet(port),
                capture=True,
            )
            url = self.tunnel.parseUrl(out.stdout or "")
            if url:
                return url
            time.sleep(2)
        log = self.session.execCode(
            self.tunnel.readLogSnippet(port),
            capture=True,
        )
        lastOutput = (log.stdout or "") + (log.stderr or "")
        self._emit(f"could not read tunnel-{port}.url; tunnel log:")
        self._emit(lastOutput.strip() or "<empty>")
        return None

    def _collectUrls(self, specs: list[ModelSpec]) -> dict[str, str]:
        urls: dict[str, str] = {}
        for spec in specs:
            self._waitHealthy(spec)
            if self.dryRun:
                urls[spec.name] = f"https://<dry-run>.loca.lt:{spec.port}"
                continue
            url = self._pollTunnelUrl(spec.port)
            urls[spec.name] = url or f"<see /content/tunnel-{spec.port}.log>"
            if url:
                self._emit(f">>> {spec.name} is live at {url} <<<")
        return urls

    def _writeUrlsFile(self, result: ProvisionResult) -> None:
        if self.dryRun:
            return
        lines = [f"{name}: {url}" for name, url in result.urls.items()]
        with open("tunnel-urls.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self._emit("tunnel URLs written to tunnel-urls.txt")

    def _printSummary(
        self,
        specs: list[ModelSpec],
        result: ProvisionResult,
    ) -> None:
        self._emit("summary")
        for spec in specs:
            url = result.urls.get(spec.name, "<pending>")
            self._emit(f"  {spec.name}: {url} (port {spec.port})")
        stop = f"colab stop -s {self.sessionName}"
        self._emit(f"session: {self.sessionName} (stop with: {stop})")
        self._emit("keeping the kernel busy so Colab does not reap the session")
        self._emit("press Ctrl+C to detach; the session keeps running")

    def _keepAlive(self) -> None:
        if self.dryRun:
            self._emit("DRY-RUN keep-alive (blocks to keep the kernel busy)")
            return
        self._emit("keep-alive: the session stays alive while this runs")
        snippet = (
            "import time\n"
            'print("keep-alive: the session stays alive while this runs")\n'
            "while True:\n"
            "    time.sleep(60)\n"
            '    print(".", flush=True)\n'
        )
        self.session.execCode(snippet, timeout=0)
