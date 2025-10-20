# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI playground export. In this notebook, you will:
# MAGIC - Author a tool-calling [MLflow's `ResponsesAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent) that uses the OpenAI client
# MAGIC - Manually test the agent's output
# MAGIC - Evaluate the agent with Mosaic AI Agent Evaluation
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC This notebook should be run on serverless or a cluster with DBR<17.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses the OpenAI SDK, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or LangGraph. To learn more, see the [Authoring Agents](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/author-agent) Databricks documentation.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC from typing import Any, Callable, Generator, Optional
# MAGIC from uuid import uuid4
# MAGIC import warnings
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from openai import OpenAI
# MAGIC from pydantic import BaseModel
# MAGIC from unitycatalog.ai.core.base import get_uc_function_client
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"
# MAGIC
# MAGIC SYSTEM_PROMPT = """You are the Data Inspector Agent. Your goal is to analyze a table in Unity Catalog and provide a concise, accurate summary of its structure.
# MAGIC
# MAGIC INPUT JSON (from the Planner Agent)
# MAGIC - uc_catalog_source: catalog to use as source
# MAGIC - uc_schema_source: schema to use as source
# MAGIC - uc_table_source: table to analyze
# MAGIC - uc_catalog_target: catalog where output will be written
# MAGIC - uc_schema_target: schema where output will be written
# MAGIC - uc_output_table: target table name for output
# MAGIC - num_records: number of records to generate
# MAGIC
# MAGIC Your work is divided into two main actions:
# MAGIC
# MAGIC (A) SOURCE VALIDATION (checks on catalog/schema/table)
# MAGIC - Purpose: ensure the specified source objects exist before any analysis.
# MAGIC - Available functions (Databricks SQL UDFs):
# MAGIC   - agentic_ai.synthia_data_agent.check_catalog_exist(catalog_name_to_find) -> 'TRUE' | 'FALSE'
# MAGIC   - agentic_ai.synthia_data_agent.check_schema_exist(catalog_name_to_find, schema_name_to_find) -> 'TRUE' | 'FALSE'
# MAGIC   - agentic_ai.synthia_data_agent.check_table_exist(catalog_name_to_find, schema_name_to_find, table_name_to_find) -> 'TRUE' | 'FALSE'
# MAGIC - Required order: catalog → schema → table.
# MAGIC - Behavior:
# MAGIC   - If a check returns 'FALSE', do not output error JSON and do not stop. Inform the user naturally and ask for a correction.
# MAGIC     Examples:
# MAGIC     - "The catalog '<uc_catalog_source>' does not exist. Would you like to provide a different one?"
# MAGIC     - "The schema '<uc_schema_source>' does not exist in '<uc_catalog_source>'. Provide another schema?"
# MAGIC     - "The table '<uc_table_source>' was not found in '<uc_catalog_source>.<uc_schema_source>'. Specify another table?"
# MAGIC   - If a check returns 'TRUE', acknowledge briefly and continue to the next check or to analysis.
# MAGIC   - Never end immediately after a tool call; always proceed to the next logical step.
# MAGIC
# MAGIC (B) TABLE ANALYSIS (metadata summary)
# MAGIC - Once all validations return 'TRUE', analyze <uc_catalog_source>.<uc_schema_source>.<uc_table_source>.
# MAGIC - Summarize, for each column:
# MAGIC   - name
# MAGIC   - data type (string, int, float, timestamp, etc.)
# MAGIC   - unique values (for categorical), or null if unavailable
# MAGIC   - min/max (for numeric), or null if unavailable
# MAGIC   - null count, or null if unavailable
# MAGIC - Present a short, human-readable summary (no step headers, no tool traces). Example:
# MAGIC   Table: financial.sales.transactions
# MAGIC   - product_id (string, 12 unique values)
# MAGIC   - price (float, range 0.99–399.99)
# MAGIC   - quantity (int, range 1–50)
# MAGIC   - date (timestamp)
# MAGIC
# MAGIC After the summary, ask:
# MAGIC "Do you want to keep all columns or select specific ones for synthetic data generation?"
# MAGIC
# MAGIC When the user confirms the selection, return ONLY the final JSON:
# MAGIC {
# MAGIC   "catalog_source": "<uc_catalog_source>",
# MAGIC   "schema_source": "<uc_schema_source>",
# MAGIC   "table_source": "<uc_table_source>",
# MAGIC   "total_columns": <int>,
# MAGIC   "selected_columns": [
# MAGIC     { "name": "<col>", "type": "<dtype>", "unique_values": <int|null>, "range": [<min|null>, <max|null>], "nulls": <int|null> }
# MAGIC   ],
# MAGIC   "excluded_columns": ["<col1>", "..."],
# MAGIC   "final_num_records": <int>
# MAGIC }
# MAGIC
# MAGIC Guidelines:
# MAGIC - Be concise and professional.
# MAGIC - Do not generate or modify data.
# MAGIC - Do not output error JSON automatically.
# MAGIC - Only produce JSON for the final confirmed configuration."""
# MAGIC
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
# MAGIC ###############################################################################
# MAGIC class ToolInfo(BaseModel):
# MAGIC     """
# MAGIC     Class representing a tool for the agent.
# MAGIC     - "name" (str): The name of the tool.
# MAGIC     - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
# MAGIC     - "exec_fn" (Callable): Function that implements the tool logic
# MAGIC     """
# MAGIC
# MAGIC     name: str
# MAGIC     spec: dict
# MAGIC     exec_fn: Callable
# MAGIC
# MAGIC
# MAGIC def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
# MAGIC     tool_spec["function"].pop("strict", None)
# MAGIC     tool_name = tool_spec["function"]["name"]
# MAGIC     udf_name = tool_name.replace("__", ".")
# MAGIC
# MAGIC     # Define a wrapper that accepts kwargs for the UC tool call,
# MAGIC     # then passes them to the UC tool execution client
# MAGIC     def exec_fn(**kwargs):
# MAGIC         function_result = uc_function_client.execute_function(udf_name, kwargs)
# MAGIC         if function_result.error is not None:
# MAGIC             return function_result.error
# MAGIC         else:
# MAGIC             return function_result.value
# MAGIC     return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)
# MAGIC
# MAGIC
# MAGIC TOOL_INFOS = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC # TODO: Add additional tools
# MAGIC UC_TOOL_NAMES = ["agentic_ai.synthia_data_agent.check_schema_exist", "agentic_ai.synthia_data_agent.check_table_exist", "agentic_ai.synthia_data_agent.check_catalog_exist"]
# MAGIC
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC uc_function_client = get_uc_function_client()
# MAGIC for tool_spec in uc_toolkit.tools:
# MAGIC     TOOL_INFOS.append(create_tool_info(tool_spec))
# MAGIC
# MAGIC
# MAGIC # Use Databricks vector search indexes as tools
# MAGIC # See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) for details
# MAGIC
# MAGIC # # (Optional) Use Databricks vector search indexes as tools
# MAGIC # # See https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools
# MAGIC # # for details
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC # # TODO: Add vector search indexes as tools or delete this block
# MAGIC # VECTOR_SEARCH_TOOLS.append(
# MAGIC #         VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # )
# MAGIC for vs_tool in VECTOR_SEARCH_TOOLS:
# MAGIC     TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))
# MAGIC
# MAGIC
# MAGIC
# MAGIC class ToolCallingAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Class representing a tool-calling Agent
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
# MAGIC         """Initializes the ToolCallingAgent with tools."""
# MAGIC         self.llm_endpoint = llm_endpoint
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = (
# MAGIC             self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         )
# MAGIC         self._tools_dict = {tool.name: tool for tool in tools}
# MAGIC
# MAGIC     def get_tool_specs(self) -> list[dict]:
# MAGIC         """Returns tool specifications in the format OpenAI expects."""
# MAGIC         return [tool_info.spec for tool_info in self._tools_dict.values()]
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name: str, args: dict) -> Any:
# MAGIC         """Executes the specified tool with the given arguments."""
# MAGIC         return self._tools_dict[tool_name].exec_fn(**args)
# MAGIC
# MAGIC     def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
# MAGIC         with warnings.catch_warnings():
# MAGIC             warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
# MAGIC             for chunk in self.model_serving_client.chat.completions.create(
# MAGIC                 model=self.llm_endpoint,
# MAGIC                 messages=to_chat_completions_input(messages),
# MAGIC                 tools=self.get_tool_specs(),
# MAGIC                 stream=True,
# MAGIC             ):
# MAGIC                 chunk_dict = chunk.to_dict()
# MAGIC                 if len(chunk_dict.get("choices", [])) > 0:
# MAGIC                     yield chunk_dict
# MAGIC
# MAGIC     def handle_tool_call(
# MAGIC         self,
# MAGIC         tool_call: dict[str, Any],
# MAGIC         messages: list[dict[str, Any]],
# MAGIC     ) -> ResponsesAgentStreamEvent:
# MAGIC         """
# MAGIC         Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
# MAGIC         """
# MAGIC         args = json.loads(tool_call["arguments"])
# MAGIC         result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
# MAGIC
# MAGIC         tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
# MAGIC         messages.append(tool_call_output)
# MAGIC         return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)
# MAGIC
# MAGIC     def call_and_run_tools(
# MAGIC         self,
# MAGIC         messages: list[dict[str, Any]],
# MAGIC         max_iter: int = 10,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         for _ in range(max_iter):
# MAGIC             last_msg = messages[-1]
# MAGIC             if last_msg.get("role", None) == "assistant":
# MAGIC                 return
# MAGIC             elif last_msg.get("type", None) == "function_call":
# MAGIC                 yield self.handle_tool_call(last_msg, messages)
# MAGIC             else:
# MAGIC                 yield from output_to_responses_items_stream(
# MAGIC                     chunks=self.call_llm(messages), aggregator=messages
# MAGIC                 )
# MAGIC
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
# MAGIC         )
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         if SYSTEM_PROMPT:
# MAGIC             messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
# MAGIC         yield from self.call_and_run_tools(messages=messages)
# MAGIC
# MAGIC
# MAGIC # Log the model using MLflow
# MAGIC mlflow.openai.autolog()
# MAGIC AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since we manually traced methods within `ResponsesAgent`, you can view the trace for each step the agent takes, with any LLM calls made via the OpenAI SDK automatically traced by autologging.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

AGENT.predict({"input": [{"role": "user", "content": "what is 4*3 in python"}]})

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What is 4*3 in Python?"}]}
):
    print(chunk.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) or leverages [external functions](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/external-connection-tools), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS, LLM_ENDPOINT_NAME
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)
for tool_name in UC_TOOL_NAMES:
    # TODO: If the UC function includes dependencies like external connection or vector search, please include them manually.
    # See the TODO in the markdown above for more information.    
    resources.append(DatabricksFunction(function_name=tool_name))

input_example = {
    "input": [
        {
            "role": "user",
            "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales,\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        pip_requirements=[
            "databricks-openai",
            "backoff",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC Evaluate your agent with one of our [predefined LLM scorers](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/predefined-judge-scorers), or try adding [custom metrics](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/custom-scorers).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness

eval_dataset = [
    {
        "inputs": {
            "input": [
                {
                    "role": "system",
                    "content": "You are the Data Inspector Agent. Your goal is to analyze a table in Unity Catalog and provide a concise, accurate summary of its structure.\n\nINPUT JSON (from the Planner Agent)\n- uc_catalog_source: catalog to use as source\n- uc_schema_source: schema to use as source\n- uc_table_source: table to analyze\n- uc_catalog_target: catalog where output will be written\n- uc_schema_target: schema where output will be written\n- uc_output_table: target table name for output\n- num_records: number of records to generate\n\nYour work is divided into two main actions:\n\n(A) SOURCE VALIDATION (checks on catalog/schema/table)\n- Purpose: ensure the specified source objects exist before any analysis.\n- Available functions (Databricks SQL UDFs):\n  - agentic_ai.synthia_data_agent.check_catalog_exist(catalog_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_schema_exist(catalog_name_to_find, schema_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_table_exist(catalog_name_to_find, schema_name_to_find, table_name_to_find) -> 'TRUE' | 'FALSE'\n- Required order: catalog → schema → table.\n- Behavior:\n  - If a check returns 'FALSE', do not output error JSON and do not stop. Inform the user naturally and ask for a correction.\n    Examples:\n    - \"The catalog '<uc_catalog_source>' does not exist. Would you like to provide a different one?\"\n    - \"The schema '<uc_schema_source>' does not exist in '<uc_catalog_source>'. Provide another schema?\"\n    - \"The table '<uc_table_source>' was not found in '<uc_catalog_source>.<uc_schema_source>'. Specify another table?\"\n  - If a check returns 'TRUE', acknowledge briefly and continue to the next check or to analysis.\n  - Never end immediately after a tool call; always proceed to the next logical step.\n\n(B) TABLE ANALYSIS (metadata summary)\n- Once all validations return 'TRUE', analyze <uc_catalog_source>.<uc_schema_source>.<uc_table_source>.\n- Summarize, for each column:\n  - name\n  - data type (string, int, float, timestamp, etc.)\n  - unique values (for categorical), or None if unavailable\n  - min/max (for numeric), or None if unavailable\n  - None count, or None if unavailable\n- Present a short, human-readable summary (no step headers, no tool traces). Example:\n  Table: financial.sales.transactions\n  - product_id (string, 12 unique values)\n  - price (float, range 0.99–399.99)\n  - quantity (int, range 1–50)\n  - date (timestamp)\n\nAfter the summary, ask:\n\"Do you want to keep all columns or select specific ones for synthetic data generation?\"\n\nWhen the user confirms the selection, return ONLY the final JSON:\n{\n  \"catalog_source\": \"<uc_catalog_source>\",\n  \"schema_source\": \"<uc_schema_source>\",\n  \"table_source\": \"<uc_table_source>\",\n  \"total_columns\": <int>,\n  \"selected_columns\": [\n    { \"name\": \"<col>\", \"type\": \"<dtype>\", \"unique_values\": <int|None>, \"range\": [<min|None>, <max|None>], \"Nones\": <int|None> }\n  ],\n  \"excluded_columns\": [\"<col1>\", \"...\"],\n  \"final_num_records\": <int>\n}\n\nGuidelines:\n- Be concise and professional.\n- Do not generate or modify data.\n- Do not output error JSON automatically.\n- Only produce JSON for the final confirmed configuration."
                },
                {
                    "role": "user",
                    "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales,\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "system",
                    "content": "You are the Data Inspector Agent. Your goal is to analyze a table in Unity Catalog and provide a concise, accurate summary of its structure.\n\nINPUT JSON (from the Planner Agent)\n- uc_catalog_source: catalog to use as source\n- uc_schema_source: schema to use as source\n- uc_table_source: table to analyze\n- uc_catalog_target: catalog where output will be written\n- uc_schema_target: schema where output will be written\n- uc_output_table: target table name for output\n- num_records: number of records to generate\n\nYour work is divided into two main actions:\n\n(A) SOURCE VALIDATION (checks on catalog/schema/table)\n- Purpose: ensure the specified source objects exist before any analysis.\n- Available functions (Databricks SQL UDFs):\n  - agentic_ai.synthia_data_agent.check_catalog_exist(catalog_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_schema_exist(catalog_name_to_find, schema_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_table_exist(catalog_name_to_find, schema_name_to_find, table_name_to_find) -> 'TRUE' | 'FALSE'\n- Required order: catalog → schema → table.\n- Behavior:\n  - If a check returns 'FALSE', do not output error JSON and do not stop. Inform the user naturally and ask for a correction.\n    Examples:\n    - \"The catalog '<uc_catalog_source>' does not exist. Would you like to provide a different one?\"\n    - \"The schema '<uc_schema_source>' does not exist in '<uc_catalog_source>'. Provide another schema?\"\n    - \"The table '<uc_table_source>' was not found in '<uc_catalog_source>.<uc_schema_source>'. Specify another table?\"\n  - If a check returns 'TRUE', acknowledge briefly and continue to the next check or to analysis.\n  - Never end immediately after a tool call; always proceed to the next logical step.\n\n(B) TABLE ANALYSIS (metadata summary)\n- Once all validations return 'TRUE', analyze <uc_catalog_source>.<uc_schema_source>.<uc_table_source>.\n- Summarize, for each column:\n  - name\n  - data type (string, int, float, timestamp, etc.)\n  - unique values (for categorical), or None if unavailable\n  - min/max (for numeric), or None if unavailable\n  - None count, or None if unavailable\n- Present a short, human-readable summary (no step headers, no tool traces). Example:\n  Table: financial.sales.transactions\n  - product_id (string, 12 unique values)\n  - price (float, range 0.99–399.99)\n  - quantity (int, range 1–50)\n  - date (timestamp)\n\nAfter the summary, ask:\n\"Do you want to keep all columns or select specific ones for synthetic data generation?\"\n\nWhen the user confirms the selection, return ONLY the final JSON:\n{\n  \"catalog_source\": \"<uc_catalog_source>\",\n  \"schema_source\": \"<uc_schema_source>\",\n  \"table_source\": \"<uc_table_source>\",\n  \"total_columns\": <int>,\n  \"selected_columns\": [\n    { \"name\": \"<col>\", \"type\": \"<dtype>\", \"unique_values\": <int|None>, \"range\": [<min|None>, <max|None>], \"Nones\": <int|None> }\n  ],\n  \"excluded_columns\": [\"<col1>\", \"...\"],\n  \"final_num_records\": <int>\n}\n\nGuidelines:\n- Be concise and professional.\n- Do not generate or modify data.\n- Do not output error JSON automatically.\n- Only produce JSON for the final confirmed configuration."
                },
                {
                    "role": "user",
                    "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales,\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
                },
                {
                    "role": "assistant",
                    "content": "I’m having trouble parsing the request—​the value for **`uc_table_source`** is missing its closing quote, and the target table field is named **`uc_table_target`** instead of the expected **`uc_output_table`**.\n\nCould you please resend the JSON with the correct table name (e.g., `\"sales\"` or another name) and the proper target‑table key? Once I have the corrected information, I’ll verify that the catalog, schema, and table exist and then provide a concise summary of the source table’s structure."
                },
                {
                    "role": "user",
                    "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales\",\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "system",
                    "content": "You are the Data Inspector Agent. Your goal is to analyze a table in Unity Catalog and provide a concise, accurate summary of its structure.\n\nINPUT JSON (from the Planner Agent)\n- uc_catalog_source: catalog to use as source\n- uc_schema_source: schema to use as source\n- uc_table_source: table to analyze\n- uc_catalog_target: catalog where output will be written\n- uc_schema_target: schema where output will be written\n- uc_output_table: target table name for output\n- num_records: number of records to generate\n\nYour work is divided into two main actions:\n\n(A) SOURCE VALIDATION (checks on catalog/schema/table)\n- Purpose: ensure the specified source objects exist before any analysis.\n- Available functions (Databricks SQL UDFs):\n  - agentic_ai.synthia_data_agent.check_catalog_exist(catalog_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_schema_exist(catalog_name_to_find, schema_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_table_exist(catalog_name_to_find, schema_name_to_find, table_name_to_find) -> 'TRUE' | 'FALSE'\n- Required order: catalog → schema → table.\n- Behavior:\n  - If a check returns 'FALSE', do not output error JSON and do not stop. Inform the user naturally and ask for a correction.\n    Examples:\n    - \"The catalog '<uc_catalog_source>' does not exist. Would you like to provide a different one?\"\n    - \"The schema '<uc_schema_source>' does not exist in '<uc_catalog_source>'. Provide another schema?\"\n    - \"The table '<uc_table_source>' was not found in '<uc_catalog_source>.<uc_schema_source>'. Specify another table?\"\n  - If a check returns 'TRUE', acknowledge briefly and continue to the next check or to analysis.\n  - Never end immediately after a tool call; always proceed to the next logical step.\n\n(B) TABLE ANALYSIS (metadata summary)\n- Once all validations return 'TRUE', analyze <uc_catalog_source>.<uc_schema_source>.<uc_table_source>.\n- Summarize, for each column:\n  - name\n  - data type (string, int, float, timestamp, etc.)\n  - unique values (for categorical), or None if unavailable\n  - min/max (for numeric), or None if unavailable\n  - None count, or None if unavailable\n- Present a short, human-readable summary (no step headers, no tool traces). Example:\n  Table: financial.sales.transactions\n  - product_id (string, 12 unique values)\n  - price (float, range 0.99–399.99)\n  - quantity (int, range 1–50)\n  - date (timestamp)\n\nAfter the summary, ask:\n\"Do you want to keep all columns or select specific ones for synthetic data generation?\"\n\nWhen the user confirms the selection, return ONLY the final JSON:\n{\n  \"catalog_source\": \"<uc_catalog_source>\",\n  \"schema_source\": \"<uc_schema_source>\",\n  \"table_source\": \"<uc_table_source>\",\n  \"total_columns\": <int>,\n  \"selected_columns\": [\n    { \"name\": \"<col>\", \"type\": \"<dtype>\", \"unique_values\": <int|None>, \"range\": [<min|None>, <max|None>], \"Nones\": <int|None> }\n  ],\n  \"excluded_columns\": [\"<col1>\", \"...\"],\n  \"final_num_records\": <int>\n}\n\nGuidelines:\n- Be concise and professional.\n- Do not generate or modify data.\n- Do not output error JSON automatically.\n- Only produce JSON for the final confirmed configuration."
                },
                {
                    "role": "user",
                    "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales,\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
                },
                {
                    "role": "assistant",
                    "content": "I’m having trouble parsing the request—​the value for **`uc_table_source`** is missing its closing quote, and the target table field is named **`uc_table_target`** instead of the expected **`uc_output_table`**.\n\nCould you please resend the JSON with the correct table name (e.g., `\"sales\"` or another name) and the proper target‑table key? Once I have the corrected information, I’ll verify that the catalog, schema, and table exist and then provide a concise summary of the source table’s structure."
                },
                {
                    "role": "user",
                    "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales\",\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
                },
                {
                    "role": "assistant",
                    "content": "Verify that the source catalog 'financial' exists before proceeding with schema and table checks.\n\n<uc_function_call>{\"id\":\"YFA2tyHbhp9BOzZPiRGQ\",\"name\":\"agentic_ai.synthia_data_agent.check_catalog_exist\",\"arguments\":\"{\\\"catalog_name_to_find\\\":\\\"financial\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"YFA2tyHbhp9BOzZPiRGQ\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"FALSE\\\"]]}\"}</uc_function_result>\n\nThe catalog **`financial`** was not found in the workspace. Could you provide a different catalog name (or confirm the correct one) so I can continue with the validation and analysis?"
                },
                {
                    "role": "user",
                    "content": "try retail customer data"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "system",
                    "content": "You are the Data Inspector Agent. Your goal is to analyze a table in Unity Catalog and provide a concise, accurate summary of its structure.\n\nINPUT JSON (from the Planner Agent)\n- uc_catalog_source: catalog to use as source\n- uc_schema_source: schema to use as source\n- uc_table_source: table to analyze\n- uc_catalog_target: catalog where output will be written\n- uc_schema_target: schema where output will be written\n- uc_output_table: target table name for output\n- num_records: number of records to generate\n\nYour work is divided into two main actions:\n\n(A) SOURCE VALIDATION (checks on catalog/schema/table)\n- Purpose: ensure the specified source objects exist before any analysis.\n- Available functions (Databricks SQL UDFs):\n  - agentic_ai.synthia_data_agent.check_catalog_exist(catalog_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_schema_exist(catalog_name_to_find, schema_name_to_find) -> 'TRUE' | 'FALSE'\n  - agentic_ai.synthia_data_agent.check_table_exist(catalog_name_to_find, schema_name_to_find, table_name_to_find) -> 'TRUE' | 'FALSE'\n- Required order: catalog → schema → table.\n- Behavior:\n  - If a check returns 'FALSE', do not output error JSON and do not stop. Inform the user naturally and ask for a correction.\n    Examples:\n    - \"The catalog '<uc_catalog_source>' does not exist. Would you like to provide a different one?\"\n    - \"The schema '<uc_schema_source>' does not exist in '<uc_catalog_source>'. Provide another schema?\"\n    - \"The table '<uc_table_source>' was not found in '<uc_catalog_source>.<uc_schema_source>'. Specify another table?\"\n  - If a check returns 'TRUE', acknowledge briefly and continue to the next check or to analysis.\n  - Never end immediately after a tool call; always proceed to the next logical step.\n\n(B) TABLE ANALYSIS (metadata summary)\n- Once all validations return 'TRUE', analyze <uc_catalog_source>.<uc_schema_source>.<uc_table_source>.\n- Summarize, for each column:\n  - name\n  - data type (string, int, float, timestamp, etc.)\n  - unique values (for categorical), or None if unavailable\n  - min/max (for numeric), or None if unavailable\n  - None count, or None if unavailable\n- Present a short, human-readable summary (no step headers, no tool traces). Example:\n  Table: financial.sales.transactions\n  - product_id (string, 12 unique values)\n  - price (float, range 0.99–399.99)\n  - quantity (int, range 1–50)\n  - date (timestamp)\n\nAfter the summary, ask:\n\"Do you want to keep all columns or select specific ones for synthetic data generation?\"\n\nWhen the user confirms the selection, return ONLY the final JSON:\n{\n  \"catalog_source\": \"<uc_catalog_source>\",\n  \"schema_source\": \"<uc_schema_source>\",\n  \"table_source\": \"<uc_table_source>\",\n  \"total_columns\": <int>,\n  \"selected_columns\": [\n    { \"name\": \"<col>\", \"type\": \"<dtype>\", \"unique_values\": <int|None>, \"range\": [<min|None>, <max|None>], \"Nones\": <int|None> }\n  ],\n  \"excluded_columns\": [\"<col1>\", \"...\"],\n  \"final_num_records\": <int>\n}\n\nGuidelines:\n- Be concise and professional.\n- Do not generate or modify data.\n- Do not output error JSON automatically.\n- Only produce JSON for the final confirmed configuration."
                },
                {
                    "role": "user",
                    "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales,\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
                },
                {
                    "role": "assistant",
                    "content": "I’m having trouble parsing the request—​the value for **`uc_table_source`** is missing its closing quote, and the target table field is named **`uc_table_target`** instead of the expected **`uc_output_table`**.\n\nCould you please resend the JSON with the correct table name (e.g., `\"sales\"` or another name) and the proper target‑table key? Once I have the corrected information, I’ll verify that the catalog, schema, and table exist and then provide a concise summary of the source table’s structure."
                },
                {
                    "role": "user",
                    "content": "{\n    \"domain\": \"retail\",\n    \"num_records\": 1000,\n    \"uc_catalog_source\": \"financial\",\n    \"uc_schema_source\": \"sales\",\n    \"uc_table_source\": \"sales\",\n    \"uc_catalog_target\": \"financial\",\n    \"uc_schema_target\": \"sales\",\n    \"uc_table_target\": \"sales\"\n\n  }"
                },
                {
                    "role": "assistant",
                    "content": "Verify that the source catalog 'financial' exists before proceeding with schema and table checks.\n\n<uc_function_call>{\"id\":\"YFA2tyHbhp9BOzZPiRGQ\",\"name\":\"agentic_ai.synthia_data_agent.check_catalog_exist\",\"arguments\":\"{\\\"catalog_name_to_find\\\":\\\"financial\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"YFA2tyHbhp9BOzZPiRGQ\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"FALSE\\\"]]}\"}</uc_function_result>\n\nThe catalog **`financial`** was not found in the workspace. Could you provide a different catalog name (or confirm the correct one) so I can continue with the validation and analysis?"
                },
                {
                    "role": "user",
                    "content": "try retail customer data"
                },
                {
                    "role": "assistant",
                    "content": "I’m happy to work with the retail customer data, but I need the exact catalog, schema, and table names you’d like to analyze. Could you provide a corrected JSON payload with those fields (e.g., `uc_catalog_source`, `uc_schema_source`, `uc_table_source`, plus the target details) so I can verify that they exist and then summarize the table’s structure?"
                },
                {
                    "role": "user",
                    "content": "can you give me the available catalog?"
                }
            ]
        },
        "expected_response": None
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety()], # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-debug#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = ""
schema = ""
model_name = ""
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/deploy-agent) for details
