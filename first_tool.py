import json
import os
from openai import OpenAI
from dotenv import load_dotenv

print("Script started")

load_dotenv()
print("Env loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("Client created")

# This is YOUR Python function — the tool
def calculate_order_cost(units, cost_per_unit, holding_days):
    procurement = units * cost_per_unit
    holding = units * cost_per_unit * 0.02 * holding_days
    total = procurement + holding
    return {
        "units": units,
        "procurement_cost": round(procurement, 2),
        "holding_cost": round(holding, 2),
        "total_cost": round(total, 2)
    }

# This tells GPT what your function does
tools = [{
    "type": "function",
    "function": {
        "name": "calculate_order_cost",
        "description": "Calculates the total cost of a supply chain order including procurement and holding costs. Use this when asked about order costs or inventory costs.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of units to order"
                },
                "cost_per_unit": {
                    "type": "number",
                    "description": "Cost per unit in GBP"
                },
                "holding_days": {
                    "type": "integer",
                    "description": "Number of days the stock will be held"
                }
            },
            "required": ["units", "cost_per_unit", "holding_days"]
        }
    }
}]

# Ask GPT a question that needs the tool
messages = [
    {
        "role": "system",
        "content": "You are a supply chain cost analyst. Use tools to calculate costs accurately. Never guess numbers."
    },
    {
        "role": "user",
        "content": "What is the total cost if Landmark Group orders 500 units at £12 each and holds them for 30 days?"
    }
]

print("Sending question to GPT...")
print("="*50)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

# Check if GPT wants to call your tool
if message.tool_calls:
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    print(f"GPT is calling your function: {tool_call.function.name}")
    print(f"With these values: {args}")
    print("="*50)
    
    # YOUR code runs the function
    result = calculate_order_cost(**args)
    print(f"Your function returned: {result}")
    print("="*50)
    
    # Send the result back to GPT
    messages.append(message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
    
    # GPT reads the result and gives final answer
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    print("GPT's final answer:")
    print(final.choices[0].message.content)