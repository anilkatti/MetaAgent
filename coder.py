from utils import _get_model, outfile
import json
import sys

sample_lla_graph_json = """
{
    "nodes": [
        {
            "name": "call_model",
            "description": "Invokes a model with the input query and figures if it already has answer or needs more information",
            "input_names": ["message"],
            "output_names": ["place_name", "final_answer"]
        },
        {
            "name": "get_weather",
            "description": "Get the current weather for the given place",
            "input_names": ["place_name"],
            "output_names": ["message"]
        }
    ],
    "edges": [
        {
            "source_node": "get_weather",
            "destination_node": "call_model"
        }
    ],
    "cond_edges": [
        {
            "source_node": "call_model",
            "destinations": [
                {
                    "node": "get_weather",
                    "condition": "continue"
                },
                {
                    "node": "END",
                    "condition": "stop"
                }
            ]
        }
    ]
}
"""

system_prompt = """

You are an expert code generator. Your task is to create a new function based on a sample function and a provided description. Follow these guidelines:

1. Carefully analyze the structure, style, and patterns of the sample function.
2. Read and understand the description of the new function to be created.
3. Generate a new function that:
    a. Maintains the same overall structure and style as the sample function.
    b. Implements the functionality described in the provided description.
    c. Uses appropriate variable names and comments that reflect the new function's purpose.
4. Ensure that the new function follows best practices for the programming language used in the sample.
5. If the description requires functionality significantly different from the sample, adapt the structure as needed while maintaining consistency where possible.

Remember, your goal is to create a function that looks and feels similar to the sample but performs the task specified in the description. Be creative but consistent in your approach. Do not provide anything else other than the python code.

Reply only in functioning python code.

"""

def coder_node(state, config):
    print(f"Generating code based on the plan ... ")
    print()

    lla_graph = state["lla_graph"]
    nodes = lla_graph["nodes"]
    edges = lla_graph["edges"]
    cond_edges = lla_graph["cond_edges"]

    model = _get_model("anthropic")

    lla_code = """
from typing import TypedDict, Literal

from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return model

workflow = StateGraph(AgentState, config_schema=GraphConfig)
    """

    for node in nodes:

        if node["name"] == "START" or node["name"] == "END":
            continue

        user_message = """
                        Sample Code:

                        system_prompt = "describe what the model is supposed to do"

                        def name(state, config):
                            messages = state["input_name"]
                            messages = [{"role": "system", "content": system_prompt}] + messages
                            model = _get_model("openai")
                            response = model.invoke(messages)
                            return {"output_name": [response.content]}

                        workflow.add_node("name", name)

                        Description:

                        """
        
        user_message = user_message + str(node)
        messages = [{"role": "system", "content": system_prompt}] + [user_message]
        model = _get_model("openai")
        response = model.invoke(messages)

        lla_code = lla_code + response.content

    for edge in edges:
        
        if edge["source_node"] == "START" or edge["destination_node"] == "END":
            continue

        user_message = """
                        Sample Code:

                        workflow.add_edge("source_node", "destination_node")

                        Description:

                        """
        
        user_message = user_message + str(edge)
        messages = [{"role": "system", "content": system_prompt}] + [user_message]
        model = _get_model("openai")
        response = model.invoke(messages)

        lla_code = lla_code + response.content

    lla_code = lla_code.replace("``````python", "")
    lla_code = lla_code.replace("```python", "")
    lla_code = lla_code.replace("```", "")

    lla_code = lla_code + "\ngraph = workflow.compile()"
    
    with open(sys.argv[2], "w") as output:
        output.write(lla_code)

    print(f"Agent code saved to {sys.argv[2]}")
    print()

    return {"lla_code": [lla_code]}
