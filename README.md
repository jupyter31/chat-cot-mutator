# chat-dsat-mutator
[Work in Progress]
## Install dependencies
- Install Python version 3.12 on your system
- Create a virtual environment in the workspace:
  - On Windows, run:
    ```powershell
    py -3.12 -m venv .venv
    ```
  - On Unix or MacOS, run:
    ```powershell
    python3.12 -m venv .venv
    ```

- Activate the virtual environment
  - On Windows, run:
    ```powershell
    .venv\Scripts\activate.bat
    ```
  - On Unix or MacOS, run:
    ```powershell
    source .venv/bin/activate.bat
    ```
- Install dependencies from `requirements.txt`
  
  ```powershell
  pip install -r requirements.txt
  ```

## Set up Foundry Launchpad token
Your Foundry Launchpad token is needed to generate a link to the Copilot Playgound Diff Tool, where you will be able to compare the original and mutated chat samples.
- Go to the [Foundry Launchpad](https://foundrylaunchpad.azurewebsites.net/) website.
- Click on the key symbol in the top right corner to see the tokens.
- Fetch and copy the `Launchpad` token.
- Create a file in the root of the repository called `foundry_token.txt` and paste the token here.

## Run the Streamlit app
To run the Streamlit app, enter the following command:

```powershell
streamlit run .\chat_dsat_mutator_app.py
```
The termimal should display a message indicating that you can now view the Streamlit app in your browser. Navigate to the `Local URL` listed below this message.

## Use the Synthetic Chat-Data Mutation Framework
1. Upload a JSONL file of chat samples, or copy and paste the chat samples into the text area.
2. Select a predefined mutation type, or write your own mutation request.
3. Choose which model you would like to perform the mutations and generate new responses from the mutated chat samples.
4. Click **Submit**.

![app-display](./images/app-display.png)
