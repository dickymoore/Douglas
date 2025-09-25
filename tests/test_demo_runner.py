import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from douglas.demo.runner import DemoRunner


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: D401, A003 - silence test server logs
        return


def test_demo_runner_executes_steps(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    script_path = tmp_path / "demo.yaml"
    script_path.write_text(
        f"""
name: test-demo
steps:
  - name: Hello world
    type: cli
    command: ["python", "-c", "print('hi')"]
  - name: Ping API
    type: api
    request:
      method: GET
      url: "http://127.0.0.1:{port}/"
  - name: GUI placeholder
    type: gui
    script: tests/gui_walkthrough.py
""",
        encoding="utf-8",
    )

    runner = DemoRunner(workspace=tmp_path / "workspace")
    try:
        report = runner.run(script_path)
    finally:
        server.shutdown()
        thread.join()

    assert report.name == "test-demo"
    assert len(report.steps) == 3
    assert report.steps[0].status == "ok"
    assert report.steps[1].status == "ok"
    assert report.steps[2].status == "ok"

    report_json = report.artifacts_dir / "report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["name"] == "test-demo"
    assert payload["steps"][1]["metadata"]["status"] == 200
