import argparse
import time
import math
from msal import PublicClientApplication
import json
import requests

class LLMClient:

    _ENDPOINT = "https://fe-26.qas.bing.net/sdf/"
    _SCOPES = ["https://substrate.office.com/llmapi/LLMAPI.dev"]
    _API_COMPLETIONS = "completions"
    _API = "chat/completions"

    def __init__(self, endpoint):
        self._app = PublicClientApplication(
            "07a99350-57ae-4197-9233-3a3dcb3ceaeb",
            authority="https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47",
            enable_broker_on_windows=True,
            enable_broker_on_mac=True,
        )
        if endpoint != None:
            LLMClient._ENDPOINT = endpoint
        LLMClient._ENDPOINT += self._API

    def send_chat_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "X-ModelType": model_name,
            "X-ScenarioGuid": "7a170a7f-3448-4fa6-9de1-57a235b71dbe",
        }

        body = str.encode(json.dumps(request))
        response = {}

        response = requests.post(
            LLMClient._ENDPOINT, data=body, headers=headers, timeout=(10, 120)
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            raise Exception(
                f"Request failed with {response.status_code}. Response: {response.text}"
            )
        else:
            raise Exception(
                f"Request failed with status code {response.status_code}. Response: {response.text}"
                )
            
    def send_batch_chat_request(self, model_name, batch_requests, batch_size=10):
        """
        Sends multiple chat completion requests as a batch to the LLM endpoint.
        
        Args:
            model_name (str): The name of the model to use
            batch_requests (list): A list of individual request objects, each containing
                                   messages and optional parameters
            batch_size (int): Maximum number of requests to send in a single batch
            
        Returns:
            list: A list of response objects, one for each request in the batch
        """
        
        # Process requests in batches of batch_size
        results = []
        total_batches = math.ceil(len(batch_requests) / batch_size)
        for i in range(0, len(batch_requests), batch_size):

            print(f"Processing batch {i // batch_size + 1} / {total_batches} with {len(batch_requests[i:i+batch_size])} requests")
            start_time = time.time()
            current_batch = batch_requests[i:i+batch_size]

            # Get the token
            token = self._get_token()

            # Populate the headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
                "X-ModelType": model_name,
                "X-ScenarioGuid": "7a170a7f-3448-4fa6-9de1-57a235b71dbe",
            }
            
            
            # For each request in the batch, ensure it has the required fields
            for req in current_batch:
                # Ensure each request has messages
                if "messages" not in req:
                    raise ValueError("Each request in the batch must contain 'messages'")
                
            # Create batch request
            # Each request in the batch needs to be processed separately in a loop
            batch_responses = []
            for request in current_batch:
                try:
                    body = str.encode(json.dumps(request))
                    response = requests.post(
                        LLMClient._ENDPOINT, 
                        data=body, 
                        headers=headers, 
                        timeout=(10, 180)  # Longer timeout for batch requests
                    )
                    
                    if response.status_code == 200:
                        batch_responses.append(response.json())
                    elif response.status_code == 429:
                        raise Exception(
                            f"Request failed with {response.status_code}. Response: {response.text}"
                        )
                    else:
                        raise Exception(
                            f"Request failed with status code {response.status_code}. Response: {response.text}"
                        )
                except Exception as e:
                    raise e
            
            # Add batch responses to results
            results.extend(batch_responses)

            end_time = time.time()
            print(f" --> Batch {i // batch_size + 1} processed in {end_time - start_time:.2f} seconds")

            # Sleep briefly between batches to avoid rate limiting
            if i + batch_size < len(batch_requests):
                time.sleep(1)

        result_contents = [r["choices"][0]["message"]["content"] for r in results]
                
        return result_contents

    def send_stream_chat_completion_request(self, model_name, request_data):
        """
        Sends a streaming chat completion request to the LLM endpoint.
        Yields partial 'content' tokens as they arrive, and eventually yields a
        final data object containing all accumulated content and function calls.
        """
        token = self._get_token()

        # Build request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-ModelType": model_name,
        }

        # Enable streaming in the request
        request_data["stream"] = True
        # Future usage: request_data["stream_options"] = { "include_usage": True }

        body = json.dumps(request_data).encode("utf-8")

        with requests.post(
            self._ENDPOINT, data=body, headers=headers, stream=True
        ) as response:
            response.raise_for_status()

            final_content = ""
            final_funcs = []
            is_final = False
            token_count = 0
            data = None

            for raw_line in response.iter_lines(decode_unicode=True):
                # Skip empty lines
                if not raw_line:
                    continue

                # Some streaming APIs prepend lines with "data: ", so handle that
                # If it's a special "[DONE]" chunk, break out
                line_str = raw_line.strip()
                if "[DONE]" in line_str:
                    break

                # Attempt to parse JSON from "line_str"
                # If it starts with e.g. "data: ", remove that prefix if needed
                # For example, if your lines come as "data: {...}".
                # Here, the original code slices off line[6:], so replicate that:
                if len(line_str) > 6:
                    line_str = line_str[6:]

                try:
                    data = json.loads(line_str)
                except Exception:
                    # Not valid JSON, skip or raise an error
                    print(f"Skipping unparseable line: {line_str}")
                    continue

                # Check for minimal structure: "choices" -> [0] -> "delta"
                if not data.get("choices") or "delta" not in data["choices"][0]:
                    # Possibly a partial or system message
                    continue

                delta_obj = data["choices"][0]["delta"]

                # 1) Process 'content' if present
                if "content" in delta_obj:
                    content = delta_obj["content"]
                    if content is not None:
                        final_content += content
                        token_count += 1
                        yield content  # Yield partial content
                    else:
                        # If content is None, optional logic
                        final_content = None

                # 2) Process 'tool_calls' if present
                if "tool_calls" in delta_obj:
                    tool_call_list = delta_obj["tool_calls"]
                    if tool_call_list:
                        func_info = tool_call_list[0]
                        if len(final_funcs) > func_info["index"]:
                            final_funcs[-1]["function"]["arguments"] += func_info[
                                "function"
                            ]["arguments"]
                        else:
                            final_funcs.append(func_info)
                        token_count += 1
                        yield ""  # Possibly yield an empty string for function calls

                # 3) If no 'content' or 'tool_calls', check "is_final"
                if "content" not in delta_obj and "tool_calls" not in delta_obj:
                    if is_final:
                        raise Exception("Unknown response; no content or tool_calls.")
                    is_final = True

            # Once stream is finished, finalize the data structure:
            if data is None:
                data = {"choices": [{"delta": {}, "message": {}}]}  # fallback structure

            # Remove "delta" from the final payload
            if data["choices"] and "delta" in data["choices"][0]:
                del data["choices"][0]["delta"]

            # Put final content into "message.content"
            if not data["choices"][0].get("message"):
                data["choices"][0]["message"] = {}
            data["choices"][0]["message"]["content"] = final_content

            # If function calls were gathered, put them in "message.tool_calls"
            if final_funcs:
                data["choices"][0]["message"]["tool_calls"] = final_funcs

            # Simulate usage info. "completion_tokens" is how many tokens we emitted
            data["usage"] = {
                "prompt_tokens": -1,  # Not tracked in this example
                "completion_tokens": token_count,
                "total_tokens": 0,  # Simplified placeholder
            }

            yield data

    def send_request(self, model_name, request, api_version=None):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "X-ModelType": model_name,
        }

        body = str.encode(json.dumps(request))
        _endpoint = LLMClient._ENDPOINT + LLMClient._API_COMPLETIONS
        _endpoint = (
            _endpoint + "?api-version=" + api_version if api_version else _endpoint
        )
        with requests.post(_endpoint, data=body, headers=headers) as response:
            response.raise_for_status()
            return response.json()

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)

        if not result:
            result = self._app.acquire_token_interactive(
                scopes=LLMClient._SCOPES,
                parent_window_handle=self._app.CONSOLE_WINDOW_HANDLE,
            )

            if "error" in result:
                raise ValueError(
                    f"Failed to acquire token. Error: {json.dumps(result, indent=4)}"
                )

        return result["access_token"]


llm_api_client = LLMClient(None)
