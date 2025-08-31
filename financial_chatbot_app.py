import streamlit as st
import pandas as pd
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Page Config ---
st.title("💰 Personal Finance Assistant")
st.caption("A chatbot that helps you with budget, savings, investments, loans, debt repayment, and taxes.")

# --- 2. Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    google_api_key = st.text_input("Google AI API Key", type="password")
    reset_button = st.button("Reset Conversation")
    test_button = st.button("Run Tool Tests")  # <-- automatic testing button

if not google_api_key:
    st.info("Please enter your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# --- Utility: Number Normalization ---
def _to_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).lower()
    s = s.replace('rp', '').replace('idr', '').replace('%', '').replace('tahun', '').replace('thn', '').replace('bln', '')
    s = re.sub(r'[^0-9,\.-]', '', s)
    if s.count('.') > 1 and s.count(',') == 0:
        s = s.replace('.', '')
    if s.count(',') == 1 and s.count('.') == 0:
        s = s.replace(',', '.')
    else:
        s = s.replace(',', '')
    try:
        return float(s)
    except:
        nums = re.findall(r'-?\d+', s)
        return float(''.join(nums)) if nums else 0.0

# --- Helper: Extract Tools Used ---
def extract_tools_used(response):
    tools_used = []
    if "messages" in response:
        for m in response["messages"]:
            if getattr(m, "tool_calls", None):  # AIMessage with tool_calls
                for t in m.tool_calls:
                    tools_used.append(t["name"])
            if m.type in ["tool", "tool_result"] and getattr(m, "name", None):
                tools_used.append(m.name)
    return list(set(tools_used))

# --- 3. Tools ---

@tool
def calculate_budget(income: float, rent: float, food: float, transport: float):
    """Calculate the budget summary, including income, expenses, and savings."""
    expenses = rent + food + transport
    savings = income - expenses
    text = f"""
**Budget Summary**
- Income: ${income:,.2f}
- Rent: ${rent:,.2f}
- Food: ${food:,.2f}
- Transport: ${transport:,.2f}
- Expenses: ${expenses:,.2f}
- Savings: ${savings:,.2f} per month
"""
    return {"text": text}

@tool
def investment_projection(amount: float, rate: float, years: int):
    """Project investment growth with compound interest."""
    amount = _to_number(amount)
    rate = _to_number(rate)
    years = int(_to_number(years))
    data = []
    for year in range(years + 1):
        value = amount * ((1 + rate/100) ** year)
        data.append({"Year": year, "Value": value})
    df = pd.DataFrame(data)
    return {
        "text": f"📈 If you invest ${amount:,.2f} at {rate}% annual return for {years} years, "
                f"your final value will be around ${df['Value'].iloc[-1]:,.2f}."
    }

@tool
def loan_calculator(principal: float, interest: float, years: int):
    """Calculate monthly loan installment and show loan balance over time."""
    principal = _to_number(principal)
    interest = _to_number(interest)
    years = int(_to_number(years))
    monthly_rate = interest / 100 / 12
    months = years * 12
    try:
        installment = principal * (monthly_rate / (1 - (1 + monthly_rate) ** -months))
    except ZeroDivisionError:
        installment = principal / months
    return {
        "text": f"💰 A loan of ${principal:,.2f} at {interest}% interest over {years} years "
                f"requires a monthly payment of about ${installment:,.2f}."
    }

@tool
def debt_repayment_calculator(principal: float, interest: float, monthly_payment: float):
    """Calculate the time needed to pay off a debt with fixed monthly payments."""
    principal = _to_number(principal)
    interest = _to_number(interest)
    monthly_payment = _to_number(monthly_payment)

    monthly_rate = interest / 100 / 12
    balance = principal
    months = 0
    while balance > 0:
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        if principal_payment <= 0:
            return {"text": "⚠️ Monthly payment too low to ever repay this debt."}
        balance -= principal_payment
        months += 1

    years = months // 12
    months_left = months % 12

    return {
        "text": f"📉 To pay off a debt of ${principal:,.2f} with {interest}% interest, "
                f"and a monthly payment of ${monthly_payment:,.2f}, it will take {years} years and {months_left} months."
    }

@tool
def tax_estimator(income: float, tax_rate: float):
    """Estimate the annual tax liability based on income and tax rate."""
    income = _to_number(income)
    tax_rate = _to_number(tax_rate)
    tax_due = income * (tax_rate / 100)
    return {
        "text": f"💸 Based on an income of ${income:,.2f} and a tax rate of {tax_rate}%, "
                f"your estimated annual tax liability is ${tax_due:,.2f}."
    }

# --- 4. Create Agent ---
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2
    )
    st.session_state.agent = create_react_agent(
        model=llm,
        tools=[calculate_budget, investment_projection, loan_calculator, debt_repayment_calculator, tax_estimator],
        prompt="""You are a strict personal finance assistant.

            CRITICAL RULES:
            - For ANY quantitative question (budget, investment, loan, debt, tax), you MUST call exactly one of the tools. NEVER compute numbers in your head.
            - Before calling a tool, normalize inputs using numeric scalars (parser already provided).
            - If the user gives multiple steps in one message, call tools sequentially (one tool call per sub-question) and explain each result briefly.
            - If inputs are missing, ask for the missing numbers.

            Tools:
            - calculate_budget(income, expenses)
            - investment_projection(amount, rate, years)
            - loan_calculator(principal, interest, years)
            - debt_repayment_calculator(principal, interest, monthly_payment)
            - tax_estimator(income, tax_rate)

            Output:
            - After tool(s) finish, summarize the result as text.
            """
    )
    st.session_state._last_key = google_api_key
    st.session_state.pop("messages", None)

# --- 5. Message History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.rerun()

# --- 6. Automatic Testing Mode ---
if test_button:
    test_cases = [
        {"role": "user", "content": "My salary is Rp20.000.000, rent Rp5.000.000, food Rp3.000.000, transport Rp2.000.000. Calculate my savings."},
        {"role": "user", "content": "If I invest Rp10.000.000 at 10% for 5 years, how much will I get?"},
        {"role": "user", "content": "I borrowed Rp100.000.000 with 12% interest over 5 years, how much is the monthly payment?"},
        {"role": "user", "content": "I have a debt of Rp50.000.000 with 10% annual interest. If I pay Rp2.000.000 per month, how long will it take to repay?"},
        {"role": "user", "content": "My annual income is Rp120.000.000 with a tax rate of 10%, how much tax should I pay?"}
    ]

    st.subheader("🔍 Tool Testing Results")
    for case in test_cases:
        st.markdown(f"**Q:** {case['content']}")
        try:
            response = st.session_state.agent.invoke({"messages": [HumanMessage(content=case["content"])]})
            if "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
            else:
                answer = "No response generated."
        except Exception as e:
            answer = f"Error: {e}"
            response = {}

        st.markdown(f"**A:** {answer}")

        tools_used = extract_tools_used(response)
        if tools_used:
            st.caption(f"✅ Tools used: {', '.join(tools_used)}")
        else:
            st.caption("❌ No tool was called")

        st.divider()

# --- 7. Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 8. Chat Input ---
prompt = st.chat_input("Ask me about your budget, investment, loan, debt, or tax...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# --- 9. Agent Response ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    try:
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        with st.spinner("Thinking..."):
            response = st.session_state.agent.invoke({"messages": messages})
            if "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
            else:
                answer = "I'm sorry, I couldn't generate a response."

    except Exception as e:
        answer = f"Error: {e}"
        response = {}

    with st.chat_message("assistant"):
        st.markdown(answer)

        tools_used = extract_tools_used(response)
        if tools_used:
            st.caption(f"✅ Tools used: {', '.join(tools_used)}")
        else:
            st.caption("❌ No tool was called")

        st.session_state.messages.append({"role": "assistant", "content": answer})