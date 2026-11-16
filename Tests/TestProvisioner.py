from ColabTensor.Provisioner import Provisioner


class TestProvisionerSelection:
    def testBothSelectsTwoSpecs(self) -> None:
        provisioner = Provisioner("s", only="both", dryRun=True)
        assert [s.name for s in provisioner._selectSpecs()] == [
            "gpt-oss-120b",
            "pplx-decider-v1.1",
        ]

    def testOnlyGptOssSelectsOneSpec(self) -> None:
        provisioner = Provisioner("s", only="gpt-oss-120b", dryRun=True)
        assert [s.name for s in provisioner._selectSpecs()] == ["gpt-oss-120b"]

    def testOnlyDeciderSelectsOneSpec(self) -> None:
        provisioner = Provisioner("s", only="pplx-decider-v1.1", dryRun=True)
        assert [s.name for s in provisioner._selectSpecs()] == [
            "pplx-decider-v1.1",
        ]


class TestProvisionerDryRun:
    def testDryRunDoesNotCallColab(self, capsys) -> None:
        provisioner = Provisioner("s", only="both", dryRun=True)
        provisioner.run()
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        assert "colab new" in out
        assert "llama.cpp" in out
        assert "localtunnel" in out
