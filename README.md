# chat-mutator
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

## Run the Streamlit app
To run the Streamlit app, enter the following command:

```powershell
streamlit run .\chat_mutator_app.py
```
The termimal should display a message indicating that you can now view the Streamlit app in your browser. Navigate to the `Local URL` listed below that message.

## Use the Synthetic Chat-Data Mutation Framework
1. Upload a JSONL file of chat samples, or copy and paste the chat samples into the text area.
2. Select a predefined mutation type, or write your own mutation request.
    - Some predefined mutation types provide the option to select a customisation.
    - The combinations of mutation type + customisation selection that are performed using an LLM will allow you to view and edit its system prompt and user query.
3. Choose which model you would like to use to perform the mutations and generate the new responses from the mutated chat samples. The default model is the recommended production model.
4. The system prompt and parameters that are used during response generation are available to view and edit.
5. Click **Submit**.

![app-display](./images/app-display.png)


## View results
It may take some time for the chat samples to be processed and for the results to appear, but once they do you will be able to do the following:
- Download all mutated chat samples with their new responses
- Download an error log to explain if any chat samples failed the mutation process.
- Generate links to the Copilot Playground Diff Tool for each chat sample to be able to see the difference highlighting.
- Run a judge which will give a score to evaluate the grounding of the new response given the mutated context.
- Click through each chat sample individually for a more detailed analysis. To see the info in the 'Extras' tab for each chat sample, the Diff Tool URLs must already be generated and the hallucination judge must be run.


![results-display](./images/results-display.png)


## Acknowledgements
Significant portion of this code was developed by Jess Peck during her internship at Microsoft.