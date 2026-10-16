import unittest
from http.client import RemoteDisconnected
from unittest.mock import patch
from urllib.error import URLError

from scripts.smoke_test import wait_for_health


class StartupRetryTests(unittest.TestCase):
    @patch("scripts.smoke_test.time.sleep")
    def test_transient_connection_errors_retry_until_ready(self, sleep):
        health = (200, {"status": "ok", "model_version": "v0.2"})
        for error in (ConnectionResetError(104, "Connection reset by peer"),
                      RemoteDisconnected("Connection closed"),
                      URLError("Connection refused"), TimeoutError("Timed out")):
            with self.subTest(error=type(error).__name__):
                sleep.reset_mock()
                with patch("scripts.smoke_test.request",
                           side_effect=[error, health]) as request:
                    self.assertEqual(wait_for_health("http://localhost:80"), health)
                    self.assertEqual(request.call_count, 2)
                    request.assert_called_with("http://localhost:80", "/health")
                    sleep.assert_called_once_with(1)

    @patch("scripts.smoke_test.time.sleep")
    def test_persistent_connection_failure_exhausts_retry_limit(self, sleep):
        with patch("scripts.smoke_test.request",
                   side_effect=ConnectionResetError("Still unavailable")) as request:
            with self.assertRaises(ConnectionResetError):
                wait_for_health("http://localhost:80", attempts=3)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    @patch("scripts.smoke_test.time.sleep")
    def test_http_response_is_returned_for_contract_validation(self, sleep):
        response = (500, {"detail": "Model failed"})
        with patch("scripts.smoke_test.request", return_value=response) as request:
            self.assertEqual(wait_for_health("http://localhost:80"), response)
            request.assert_called_once()
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
