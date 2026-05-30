import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from typing import Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
 
load_dotenv()
app = Flask(__name__)
 
 
# --- LANGCHAIN TOOLS ---
 
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """Fetches the currency conversion rate between two currencies."""
    url = f"https://api.frankfurter.app/latest?from={base_currency.upper()}&to={target_currency.upper()}"
    response = requests.get(url)
    return response.json()
 
 
@tool
def convert(base_currency_val: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Calculates the final converted value given an amount and a conversion rate."""
    return base_currency_val * conversion_rate
 
 
# FIX 1: Correct model name — "gemini-3.5-flash" does not exist.
# Use "gemini-2.0-flash" or "gemini-1.5-flash".
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b", google_api_key=os.getenv("GOOGLE_API_KEY"))
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])
 
 
def safe_content(content) -> str:
    """
    FIX 2: Gemini can return content as a list of blocks, not a plain string.
    This causes [object Object] in the browser. We normalize it here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)
    return str(content)
 
 
# --- FLASK ROUTES ---
 
@app.route('/')
def home():
    return render_template('index.html')
 
 
@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_query = request.json.get('message', '').strip()
        if not user_query:
            return jsonify({"response": "Please enter a conversion query."}), 400
 
        messages = [HumanMessage(content=user_query)]
 
        # --- Turn 1: LLM decides which tool to call first ---
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
 
        conv_rate = None
 
        if ai_msg.tool_calls:
            for tc in ai_msg.tool_calls:
                if tc['name'] == 'get_conversion_factor':
                    res = get_conversion_factor.invoke(tc['args'])
                    messages.append(ToolMessage(content=json.dumps(res), tool_call_id=tc['id']))
 
                    rates = res.get('rates', {})
                    target = tc['args'].get('target_currency', '').upper()
                    conv_rate = rates.get(target)
 
            # --- Turn 2: LLM calls convert() with the fetched rate ---
            ai_msg_2 = llm_with_tools.invoke(messages)
            messages.append(ai_msg_2)
 
            if ai_msg_2.tool_calls:
                for tc in ai_msg_2.tool_calls:
                    if tc['name'] == 'convert':
                        # FIX 3: Inject conv_rate only if we have it; fall back to
                        # whatever the LLM provided so the tool still runs.
                        if conv_rate is not None:
                            tc['args']['conversion_rate'] = conv_rate
                        res = convert.invoke(tc['args'])
                        messages.append(ToolMessage(content=str(res), tool_call_id=tc['id']))
 
        # --- Turn 3: LLM composes the final human-readable answer ---
        final_res = llm_with_tools.invoke(messages)
 
        # FIX 2 (applied): normalize content before returning
        return jsonify({"response": safe_content(final_res.content)})
 
    except Exception as e:
        print(f"CRASH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"Backend error: {str(e)}"}), 500
 
 
if __name__ == "__main__":
    app.run(debug=True)