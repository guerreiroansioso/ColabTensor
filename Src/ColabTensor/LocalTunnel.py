"""localtunnel exposure for the VM's localhost ports."""

import re

from ColabTensor.Backends import detachedSnippet

TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.loca\.lt")


class LocalTunnel:
    """Launches and reads localtunnel tunnels running on the VM."""

    def installSnippet(self) -> str:
        return (
            "import subprocess\n"
            "subprocess.run(\n"
            "    ['npm', 'install', '--prefix', '/content', 'localtunnel'],\n"
            "    check=True,\n"
            ")\n"
            'print("localtunnel installed")\n'
        )

    def launchSnippet(self, port: int) -> str:
        script = (
            "const lt = require('localtunnel');"
            f"lt({{port: {port}}}).then(t => {{"
            f"require('fs').writeFileSync('/content/tunnel-{port}.url', t.url);"
            "console.log('your url is:', t.url);"
            "}).catch(e => { console.error(e); process.exit(1); });"
        )
        argv = ["node", "-e", script]
        return detachedSnippet(f"tunnel-{port}.log", argv)

    def readUrlSnippet(self, port: int) -> str:
        return f'print(open("/content/tunnel-{port}.url").read())\n'

    def readLogSnippet(self, port: int) -> str:
        return f'print(open("/content/tunnel-{port}.log").read())\n'

    def parseUrl(self, logText: str) -> str | None:
        match = TUNNEL_URL_PATTERN.search(logText)
        return match.group(0) if match else None
