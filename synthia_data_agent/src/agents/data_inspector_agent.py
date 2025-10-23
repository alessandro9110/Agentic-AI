import yaml
import json
from databricks_langchain import  ChatDatabricks, DatabricksFunctionClient, UCFunctionToolkit, set_uc_function_client

from typing import Any, Generator, Literal

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import AgentExecutor, create_tool_calling_agent

from pydantic import BaseModel
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages, AnyMessage

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.graph import MessagesState 
from langgraph.checkpoint.memory import MemorySaver

# Custom imports
from configs import variables
from prompts import data_inspector


from langchain.agents import AgentExecutor, create_tool_calling_agent

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, Dict, List, Optional
from langgraph.graph.message import add_messages, AnyMessage


#######################################################
# Configure the Foundation Model
#######################################################
chat_model = ChatDatabricks(endpoint=variables.LLM_ENDPOINT_NAME)


#######################################################
# Configure Tools
#######################################################
# Add all the Unity Catalog functions explicitly for data inspection
uc_function_names = [
    f"{variables.CATALOG_NAME}.{variables.SCHEMA_NAME}.check_catalog_exist",
    f"{variables.CATALOG_NAME}.{variables.SCHEMA_NAME}.check_schema_exist", 
    f"{variables.CATALOG_NAME}.{variables.SCHEMA_NAME}.check_table_exist",
    f"{variables.CATALOG_NAME}.{variables.SCHEMA_NAME}.get_table_columns",
    #f"{variables.CATALOG_NAME}.{variables.SCHEMA_NAME}.get_column_statistics",
    #f"{variables.CATALOG_NAME}.{variables.SCHEMA_NAME}.get_table_summary"
]


# assign function to UCFunctionToolkit
toolkit = UCFunctionToolkit( function_names=uc_function_names)
tools_uc = toolkit.tools
llm_with_tools = chat_model.bind_tools(tools_uc)

# Define the state
class DataInspectorState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[AnyMessage], add_messages]

# Create system message for Agents
prompt = SystemMessage(content=data_inspector.system_prompt)

# Create an Agent
def data_inspector_agent(state: DataInspectorState):
    response = [llm_with_tools.invoke([prompt] + state["messages"])]
    last_message = response[-1]
    return {"messages": response}


class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def _langchain_to_responses(self, message: BaseMessage) -> list[dict[str, Any]]:
        "Convert from ChatCompletion dict to Responses output item dictionaries. Ignore user and human messages"
        message = message.model_dump()
        role = message["type"]
        output = []
        if role == "ai":
            if message.get("content"):
                output.append(
                    self.create_text_output_item(
                        text=message["content"],
                        id=message.get("id") or str(uuid4()),
                    )
                )
            if tool_calls := message.get("tool_calls"):
                output.extend(
                    [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                        for tool_call in tool_calls
                    ]
                )

        elif role == "tool":
            output.append(
                self.create_function_call_output_item(
                    call_id=message["tool_call_id"],
                    output=message["content"],
                )
            )
        elif role == "user" or "human":
            pass
        return output

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest,) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
        first_name = True
        seen_ids = set()

        for event_name, events in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates"]):
            if event_name == "updates":
                if not first_name:
                    node_name = tuple(events.keys())[0]  # assumes one name per node
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(
                            text=f"<name>{node_name}</name>",
                            id=str(uuid4()),
                        ),
                    )
                for node_data in events.values():
                    for msg in node_data["messages"]:
                        if msg.id not in seen_ids:
                            print(msg.id, msg)
                            seen_ids.add(msg.id)
                            for item in self._langchain_to_responses(msg):
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done", item=item
                                )
            first_name = False


# --- nodes
builder = StateGraph(DataInspectorState)
builder.add_node("data_inspector", data_inspector_agent)
builder.add_node("tools", ToolNode(tools_uc))

# --- edges
builder.add_edge(START, "data_inspector")
builder.add_edge("tools", "data_inspector")   # ritorno dopo l'esecuzione tool

def tools_condition(state: DataInspectorState) -> str:
    last = state["messages"][-1]
    # LangChain msg objects: assistant msgs hanno .tool_calls o dict key "tool_calls"

    # Se contiene tool_calls → chiama tools
    if getattr(last, "tool_calls", None):
        return "tools"

    # Altrimenti fine
    return END              # nessun tool-call -> termina

builder.add_conditional_edges("data_inspector", tools_condition)
# IMPORTANT: niente builder.add_edge("data_inspector", "tools") e niente edge extra verso END

memory = MemorySaver()
react_graph = builder.compile()

mlflow.langchain.autolog()
AGENT = LangGraphResponsesAgent(react_graph)
mlflow.models.set_model(AGENT)
