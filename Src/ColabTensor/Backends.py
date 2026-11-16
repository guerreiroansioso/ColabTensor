"""Serving backend: llama.cpp's llama-server for both models.

Both models are served by llama-server (OpenAI-compatible). gpt-oss-120b
uses unsloth's Q4_K_M GGUF; pplx-decider-v1.1 uses TurinTech's Artemis
Q4_K_S GGUF (based on pplx-decider-v1.1-27b, protocol v1).
"""

from typing import Protocol

from ColabTensor.ModelSpec import ModelSpec

LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b12441/llama-b12441-bin-ubuntu-cuda-12.8-x64.tar.gz"
)
LLAMA_CPP_CUDART_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b12441/cudart-llama-b12441-bin-ubuntu-cuda-12.8-x64.tar.gz"
)


class Backend(Protocol):
    """How one model is downloaded, launched and health-checked."""

    def installPackages(self) -> list[str]: ...

    def downloadSnippet(self, spec: ModelSpec) -> str: ...

    def launchSnippet(self, spec: ModelSpec) -> str: ...

    def healthPath(self) -> str: ...


def detachedSnippet(
    logName: str,
    argv: list[str],
    cwd: str = "/content",
) -> str:
    argvLiteral = ", ".join(f'"{a}"' for a in argv)
    return (
        "import subprocess\n"
        f'log = open("/content/{logName}", "w")\n'
        "subprocess.Popen(\n"
        f"    [{argvLiteral}],\n"
        f'    cwd="{cwd}",\n'
        "    stdout=log,\n"
        "    stderr=subprocess.STDOUT,\n"
        "    start_new_session=True,\n"
        ")\n"
        'print("launched")\n'
    )


class LlamaCppBackend:
    """Serves a GGUF model with llama-server."""

    def installPackages(self) -> list[str]:
        return ["huggingface-hub"]

    def downloadSnippet(self, spec: ModelSpec) -> str:
        patterns = ", ".join(f'"{p}"' for p in spec.hfInclude)
        return (
            "from huggingface_hub import snapshot_download\n"
            f'snapshot_download(repo_id="{spec.hfRepo}", '
            f"allow_patterns=[{patterns}], "
            f'local_dir="{spec.localDir}")\n'
            'print("downloaded")\n'
        )

    def launchSnippet(self, spec: ModelSpec) -> str:
        modelPath = f"{spec.localDir}/{spec.modelFile}"
        argv = [
            "-m",
            modelPath,
            "--port",
            str(spec.port),
            "-c",
            str(spec.contextSize),
            *spec.extraArgs,
            "--host",
            "0.0.0.0",
        ]
        argvLiteral = ", ".join(f'"{a}"' for a in argv)
        return (
            "import glob\n"
            "import os\n"
            "import subprocess\n"
            'server = open("/content/llama_server_path.txt").read().strip()\n'
            "libDirs = {os.path.dirname(server)}\n"
            "libDirs.update(\n"
            "    os.path.dirname(p)\n"
            '    for p in glob.glob("/content/llama/**/libcudart.so*", '
            "recursive=True)\n"
            ")\n"
            "env = dict(os.environ)\n"
            'env["LD_LIBRARY_PATH"] = ":".join(libDirs) + ":" + '
            'env.get("LD_LIBRARY_PATH", "")\n'
            f'log = open("/content/{spec.name}.log", "w")\n'
            "subprocess.Popen(\n"
            f"    [server, {argvLiteral}],\n"
            "    env=env,\n"
            "    stdout=log,\n"
            "    stderr=subprocess.STDOUT,\n"
            "    start_new_session=True,\n"
            ")\n"
            'print("launched")\n'
        )

    def healthPath(self) -> str:
        return "/health"


BACKENDS: dict[str, Backend] = {
    "llama-server": LlamaCppBackend(),
}
