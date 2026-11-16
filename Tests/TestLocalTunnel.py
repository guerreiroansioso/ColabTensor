from ColabTensor.LocalTunnel import LocalTunnel


class TestLocalTunnel:
    def testLaunchSnippetUsesNodeLocaltunnel(self) -> None:
        snippet = LocalTunnel().launchSnippet(8000)
        assert "localtunnel" in snippet
        assert "8000" in snippet
        assert "start_new_session=True" in snippet

    def testInstallSnippetInstallsLocaltunnel(self) -> None:
        snippet = LocalTunnel().installSnippet()
        assert "npm" in snippet
        assert "localtunnel" in snippet

    def testReadUrlSnippetReadsUrlFile(self) -> None:
        snippet = LocalTunnel().readUrlSnippet(8000)
        assert "tunnel-8000.url" in snippet

    def testReadLogSnippetReadsLog(self) -> None:
        snippet = LocalTunnel().readLogSnippet(8000)
        assert "tunnel-8000.log" in snippet

    def testParseUrlExtractsLocaLtUrl(self) -> None:
        log = "your url is: https://cool-name-123.loca.lt\n"
        assert LocalTunnel().parseUrl(log) == "https://cool-name-123.loca.lt"

    def testParseUrlReturnsNoneWhenMissing(self) -> None:
        assert LocalTunnel().parseUrl("no url here") is None
