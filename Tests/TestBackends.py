from ColabTensor.Backends import BACKENDS, LlamaCppBackend
from ColabTensor.ModelSpec import GPT_OSS_120B, PPLX_DECIDER


class TestLlamaCppBackend:
    def testInstallsHuggingfaceHub(self) -> None:
        assert "huggingface-hub" in LlamaCppBackend().installPackages()

    def testDownloadSnippetUsesSnapshotDownload(self) -> None:
        snippet = LlamaCppBackend().downloadSnippet(GPT_OSS_120B)
        assert "snapshot_download" in snippet
        assert GPT_OSS_120B.hfRepo in snippet
        assert "Q4_K_M/*" in snippet
        assert GPT_OSS_120B.localDir in snippet

    def testLaunchSnippetRunsLlamaServer(self) -> None:
        snippet = LlamaCppBackend().launchSnippet(GPT_OSS_120B)
        assert "llama_server_path.txt" in snippet
        assert GPT_OSS_120B.modelFile in snippet
        assert str(GPT_OSS_120B.port) in snippet
        assert "start_new_session=True" in snippet

    def testDeciderLaunchUsesNoMmproj(self) -> None:
        snippet = LlamaCppBackend().launchSnippet(PPLX_DECIDER)
        assert "--no-mmproj" in snippet
        assert PPLX_DECIDER.modelFile in snippet

    def testHealthPath(self) -> None:
        assert LlamaCppBackend().healthPath() == "/health"


class TestBackendRegistry:
    def testRegistryHasLlamaServer(self) -> None:
        assert set(BACKENDS) == {"llama-server"}
