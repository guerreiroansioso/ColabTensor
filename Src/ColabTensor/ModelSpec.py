"""Model registry: the two Q4 GGUF checkpoints served on the G4.

gpt-oss-120b uses unsloth's Q4_K_M GGUF (2 shards). pplx-decider-v1.1 uses
TurinTech's Artemis Q4_K_S GGUF, based on pplx-decider-v1.1-27b (protocol
v1) — the only Q4 llama.cpp-compatible decider.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one served model."""

    name: str
    hfRepo: str
    hfInclude: tuple[str, ...]
    localDir: str
    modelFile: str
    port: int
    contextSize: int
    extraArgs: tuple[str, ...]


GPT_OSS_120B = ModelSpec(
    name="gpt-oss-120b",
    hfRepo="unsloth/gpt-oss-120b-GGUF",
    hfInclude=("Q4_K_M/*",),
    localDir="/content/models/gpt-oss-120b",
    modelFile="Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf",
    port=8000,
    contextSize=12000,
    extraArgs=("-ngl", "99"),
)

PPLX_DECIDER = ModelSpec(
    name="pplx-decider-v1.1",
    hfRepo="turintech/pplx-decider-27B-Artemis-GGUF",
    hfInclude=("pplx-decider-27B-Artemis-Q4_K_S.gguf",),
    localDir="/content/models/pplx-decider",
    modelFile="pplx-decider-27B-Artemis-Q4_K_S.gguf",
    port=8001,
    contextSize=16384,
    extraArgs=("-ub", "4096", "-b", "4096", "--no-mmproj", "-ngl", "99"),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    GPT_OSS_120B.name: GPT_OSS_120B,
    PPLX_DECIDER.name: PPLX_DECIDER,
}

DEFAULT_MODELS: tuple[str, ...] = (GPT_OSS_120B.name, PPLX_DECIDER.name)
