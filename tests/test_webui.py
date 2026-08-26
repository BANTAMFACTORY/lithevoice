"""HTTP contract tests for the localhost dashboard."""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from webui import Bus, _make_handler


class WebUiHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bus = Bus()
        cls.commands = []
        cls.bus.on_command = cls.commands.append
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(cls.bus))
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, payload, content_type="application/json", origin=None):
        headers = {"Content-Type": content_type}
        if origin is not None:
            headers["Origin"] = origin
        return urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{self.port}/control",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            ),
            timeout=3,
        )

    def test_dashboard_serves_lithevoice_ui(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=3) as response:
            body = response.read().decode()
        self.assertIn("<title>LitheVoice</title>", body)
        self.assertIn('id="state-name"', body)

    def test_valid_same_origin_json_control_reaches_bus(self):
        before = len(self.commands)
        with self.request(
            {"action": "speed", "value": 1.1},
            origin=f"http://127.0.0.1:{self.port}",
        ) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.commands[before], {"action": "speed", "value": 1.1})

    def test_non_json_control_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request({"action": "shutdown", "value": None}, content_type="text/plain")
        self.assertEqual(caught.exception.code, 415)

    def test_foreign_origin_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                {"action": "shutdown", "value": None},
                origin="https://example.com",
            )
        self.assertEqual(caught.exception.code, 403)

    def test_malformed_origin_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                {"action": "shutdown", "value": None},
                origin="http://localhost:not-a-port",
            )
        self.assertEqual(caught.exception.code, 403)

    def test_non_finite_speed_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request({"action": "speed", "value": float("nan")})
        self.assertEqual(caught.exception.code, 400)

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request({"action": "launch", "value": None})
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
