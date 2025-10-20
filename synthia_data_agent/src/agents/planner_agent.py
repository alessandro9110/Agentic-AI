import yaml
from databricks_langchain import  ChatDatabricks
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

from configs import variables
from prompts import planner


# Define LLm endpoint
chat_model = ChatDatabricks(endpoint=variables.LLM_ENDPOINT_NAME)

# Set State
class PlannerState(TypedDict):
    """The state of the agent."""
    messages: Annotated[list[AnyMessage], add_messages]

# Create system message for Agents
prompt = SystemMessage(content=planner.system_prompt)

# Create an Agent
def planner_agent(state: PlannerState):
    response = [chat_model.invoke([prompt] + state["messages"])]
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
                content = message["content"]

                # --- 🔧 Sanitize content to always be a string
                if isinstance(content, list):
                    # concatena eventuali blocchi di tipo text/reasoning
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                elif isinstance(content, dict):
                    # se è un singolo dict, estrai 'text' o converti in stringa
                    content = content.get("text", str(content))
                elif not isinstance(content, str):
                    # fallback finale
                    content = str(content)

                output.append(
                    self.create_text_output_item(
                        text=content.strip(),
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


# Build graph
builder = StateGraph(PlannerState)
builder.add_node("planner", planner_agent)

# Logic
builder.add_edge(START, "planner")


# Add memory. In this way we can save the state of the agent and mantain the memory to save the chat history. 
# This can be extended with external memory with Lakebase
memory = MemorySaver()
react_graph = builder.compile()

mlflow.langchain.autolog()
AGENT = LangGraphResponsesAgent(react_graph)
mlflow.models.set_model(AGENT)
