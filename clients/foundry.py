from pathlib import Path
import time
from typing import NamedTuple
import uuid
import httpx

_USER_OID = "19b50cd6-7371-420f-b506-6c3bc9c02f4a"  # dmstrati
_TENANT_ID = "72f988bf-86f1-41af-91ab-2d7cd011db47"
_USERNAME = "dmstrati"

_FOUNDRY_BASE_URL = "https://m365playground.prod.substrateai.microsoft.net"


def _get_foundry_refresh_token() -> str:
    # grab it from the browser
    refresh_token = (
        open(Path(__file__).parent.parent.parent / "foundry_refresh_token.txt.user")
        .read()
        .strip()
    )
    return refresh_token


class FoundryAccessToken(NamedTuple):
    """Represents an OAuth access token."""

    token: str
    expires_on: int


def get_foundry_token(refresh_token: str) -> FoundryAccessToken:
    now = int(time.time())
    data = {
        # 'resource': 'api://cf5c3336-f1f1-435e-9d31-b32a6638ba15/access '
        #             'api://cf5c3336-f1f1-435e-9d31-b32a6638ba15/FoundryToolkit.Use',
        "client_id": "cf5c3336-f1f1-435e-9d31-b32a6638ba15",  # Foundry App id
        "client_info": "1",
        "client-request-id": str(uuid.uuid4()),
        "grant_type": "refresh_token",
        # 'username': username,
        "refresh_token": refresh_token,
        "scope": "api://cf5c3336-f1f1-435e-9d31-b32a6638ba15/FoundryToolkit.Use openid offline_access",
        "X-AnchorMailbox": f"Oid:{_USER_OID}@{_TENANT_ID}",
    }
    headers = {
        "Accept": "*/*",
        # 'Accept': 'application/json',
        "content-type": "application/x-www-form-urlencoded;charset=utf-8",
        "Origin": _FOUNDRY_BASE_URL,
        "Referer": f"{_FOUNDRY_BASE_URL}/",
    }
    #print("Requesting Foundry access token...")
    response = httpx.post(
        f"https://login.microsoftonline.com/{_TENANT_ID}/oauth2/v2.0/token",
        data=data,
        headers=headers,
    )
    #print(response.status_code)
    #print(response.headers)
    #print(response.text)
    result = response.json()
    access_token = result["access_token"]
    return FoundryAccessToken(access_token, now + int(result["expires_in"]))


class FoundryClient:
    access_token: FoundryAccessToken

    def __init__(self):
        self.access_token = None
        self.client = httpx.Client()

    def _get_token(self) -> str:
        # if self.access_token is not None:
        #     current_time = int(time.time())
        #     if current_time < self.access_token.expires_on:
        #         return self.access_token.token

        # refresh_token = _get_foundry_refresh_token()
        # self.access_token = get_foundry_token(refresh_token)
        # return self.access_token.token

        with open(Path(__file__).parent.parent / "token.txt") as f:
            token = f.read().strip()
        
        return token

    def save_diff(
        self,
        control: str,
        treatment: str,
        control_label: str,
        treatment_label: str,
        language="json",
        diff_guid: str = None,
    ) -> str:
        diff_guid = diff_guid or str(uuid.uuid4())
        diff_data = {
            "id": diff_guid,
            "language": language,
            "left": control,
            "right": treatment,
            "leftName": control_label,
            "rightName": treatment_label,
            "owners": [_USERNAME],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_token()}",
            "Accept": "application/json",
            "Origin": _FOUNDRY_BASE_URL,
            "Referer": f"{_FOUNDRY_BASE_URL}/",
        }

        #print(f"Saving diff {control_label} vs {treatment_label}...")
        response = self.client.post(
            f"{_FOUNDRY_BASE_URL}/api/v2/data/diff/save",
            json=diff_data,
            headers=headers,
        )
        assert (
            response.status_code == 200
        ), f"Failed to save diff: {response.status_code} - {response.text}"

        final_diff_url = self.get_diff_url(diff_guid)
        #print(f"Diff saved: {final_diff_url}")
        return final_diff_url

    def get_diff_url(self, diff_guid: str) -> str:
        return f"{_FOUNDRY_BASE_URL}/diffTool?diff={diff_guid}"


foundry_client = FoundryClient()
