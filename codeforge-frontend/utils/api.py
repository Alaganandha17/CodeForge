# utils/api.py

import os
import requests

BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


class APIClient:

    def health_check(self):
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def optimize(self, code: str):
        try:
            response = requests.post(
                f"{BASE_URL}/optimize",
                json={"code": code},
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            return {
                "status": "ERROR",
                "error": "Cannot connect to backend. Make sure the backend server is running.",
                "message": "Backend unavailable"
            }
        except requests.exceptions.Timeout:
            return {
                "status": "ERROR",
                "error": "Request timed out. The code may be too complex to optimize.",
                "message": "Request timeout"
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "message": str(e)
            }

    def optimize_rules_only(self, code: str):
        try:
            response = requests.post(
                f"{BASE_URL}/optimize-rules-only",
                json={"code": code},
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            return {
                "status": "ERROR",
                "error": "Cannot connect to backend. Make sure the backend server is running.",
                "message": "Backend unavailable"
            }
        except requests.exceptions.Timeout:
            return {
                "status": "ERROR",
                "error": "Request timed out. The code may be too complex to benchmark.",
                "message": "Request timeout"
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "message": str(e)
            }

    def upload_file(self, file_bytes, filename: str):
        try:
            files = {
                "file": (filename, file_bytes, "text/x-python")
            }

            response = requests.post(
                f"{BASE_URL}/upload",
                files=files,
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            return {
                "status": "ERROR",
                "error": "Cannot connect to backend. Make sure the backend server is running.",
                "message": "Backend unavailable"
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "message": str(e)
            }


api_client = APIClient()
