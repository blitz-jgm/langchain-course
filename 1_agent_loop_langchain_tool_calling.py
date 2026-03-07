from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "claude-haiku-4-5"


# --- Tools (Langchain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    print(f"Getting price for {product}...")
    prices = {
        "laptop": 1299.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "keyboard": 49.49,
    }
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier."""
    print(f"Applying {discount_tier} discount to price {price}...")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
        "platinum": 30,
    }
    discount_rate = discount_percentages.get(discount_tier, 0)
    discounted_price = price * (1 - discount_rate / 100)
    return round(discounted_price, 2)


# --- Agent Loop ---


@traceable(name="Langchain agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]

    tools_dict = {t.name: t for t in tools}
    
    llm = init_chat_model(MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "Assume the product and discount information is in the tools"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration}...")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        print(f"Model response:\n{ai_message}\n")
        print(f"Model tool calls:\n{ai_message.tool_calls}\n")

        # If no tool calls, end the loop
        if not tool_calls:
            print(f"\nModel did not call any tools. Ending loop with Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call in this iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"Tool call detected: {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            print(f"Error: Tool '{tool_name}' not found. Ending loop.")
            raise ValueError(f"Tool '{tool_name}' not found")
    
        observation = tool_to_use.invoke(tool_args)

        print(f"Observation from tool call: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello from langchain-course!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")