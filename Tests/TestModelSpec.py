from ColabTensor.ModelSpec import (
    DEFAULT_MODELS,
    GPT_OSS_120B,
    MODEL_REGISTRY,
    PPLX_DECIDER,
)

GPT_OSS_PORT = 8000
DECIDER_PORT = 8001


class TestModelSpec:
    def testRegistryContainsBothModels(self) -> None:
        assert set(MODEL_REGISTRY) == {"gpt-oss-120b", "pplx-decider-v1.1"}

    def testDefaultModelsAreBoth(self) -> None:
        assert DEFAULT_MODELS == ("gpt-oss-120b", "pplx-decider-v1.1")

    def testGptOssIsQ4K_M(self) -> None:
        assert "Q4_K_M" in GPT_OSS_120B.modelFile
        assert GPT_OSS_120B.port == GPT_OSS_PORT
        assert "unsloth" in GPT_OSS_120B.hfRepo

    def testDeciderIsQ4K_S(self) -> None:
        assert "Q4_K_S" in PPLX_DECIDER.modelFile
        assert PPLX_DECIDER.port == DECIDER_PORT
        assert "turintech" in PPLX_DECIDER.hfRepo

    def testPortsAreDistinct(self) -> None:
        assert GPT_OSS_120B.port != PPLX_DECIDER.port
