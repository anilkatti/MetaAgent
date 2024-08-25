

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
    ```python
system_prompt = "parse the user's weather request"

def parse_request(state, config):
    messages = state["initial_request"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"parsed_request": [response.content]}

workflow.add_node("parse_request", parse_request)
``````python
system_prompt = "Extract or request location information"

def get_location(state, config):
    messages = state["parsed_request"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"location": [response.content]}

workflow.add_node("get_location", get_location)
``````python
system_prompt = "fetch weather data for the given location"

def fetch_weather_data(state, config):
    location = state["location"]
    messages = [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": location}]
    model = _get_model("weather_api")
    response = model.invoke(messages)
    return {"weather_data": [response.content]}

workflow.add_node("fetch_weather_data", fetch_weather_data)
``````python
system_prompt = "Generate a human-readable weather report"

def generate_response(state, config):
    messages = state["weather_data"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = _get_model("openai")
    response = model.invoke(messages)
    return {"weather_report": [response.content]}

workflow.add_node("generate_response", generate_response)
``````python
workflow.add_edge("START", "parse_request")
``````python
workflow.add_edge("parse_request", "get_location")
``````python
workflow.add_edge("get_location", "fetch_weather_data")
``````python
workflow.add_edge("fetch_weather_data", "generate_response")
``````python
workflow.add_edge("generate_response", "END")
```