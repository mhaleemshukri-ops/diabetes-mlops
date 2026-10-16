"""Check the assignment's API contract against a running container."""

import argparse
import json
import math
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base_url + path, data=data,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def wait_for_health(base_url, attempts=30):
    for attempt in range(attempts):
        try:
            return request(base_url, "/health")
        except (URLError, OSError, HTTPException):
            # Docker can accept connections before the server is ready, then
            # reset or close them without returning an HTTP response.
            if attempt == attempts - 1:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:80")
    parser.add_argument("--expected-version", required=True, choices=("v0.1", "v0.2"))
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    status, health = wait_for_health(base_url)
    assert status == 200, (status, health)
    assert health == {"status": "ok", "model_version": args.expected_version}, health

    payload = {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
               "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
               "s5": 0.02, "s6": -0.001}
    status, result = request(base_url, "/predict", payload)
    assert status == 200, (status, result)
    assert set(result) == {"prediction"}, result
    assert type(result["prediction"]) in (int, float), result
    assert math.isfinite(result["prediction"]), result

    del payload["s6"]
    status, error = request(base_url, "/predict", payload)
    assert status == 422, (status, error)
    assert any(item.get("loc") == ["body", "s6"]
               for item in error.get("detail", [])), error
    print(f"Container smoke tests passed for {args.expected_version}")


if __name__ == "__main__":
    main()
