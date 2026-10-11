import fastai as f

tools = [{'type': 'function', 'function': {'name': 'browse', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}]
try:
    result = f.chat(
        [{'role': 'user', 'content': 'search for python'}],
        tools=tools,
        tool_choice='auto',
        max_tokens=100
    )
    print('Result:', result)
except Exception as e:
    print('Error:', type(e).__name__, e)
    import traceback
    traceback.print_exc()
