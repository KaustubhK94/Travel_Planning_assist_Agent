import streamlit as st
import time
from travel_agent import build_agent, StreamlitCallbackHandler

# ---- Build the agent ----
if "agent" not in st.session_state:
    st.session_state.agent = build_agent(verbose=False)
agent = st.session_state.agent

# ---- Streamlit Page Config ----
st.set_page_config(page_title="TripMate - Travel Assistant ", page_icon="✈️", layout="wide")
# st.title("✈️ TripMate - Your AI Travel Planner")
st.markdown(
    """
    <h1 style='font-size: 28px; margin-bottom: 20px;'>✈️ TripMate - Your AI Travel Planner</h1>
    """,
    unsafe_allow_html=True
)

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    body {
        background-color: #ECECEC;
    }
    .chat-container {
        padding: 10px;
    }
    .msg-container {
        display: flex;
        margin: 8px 0;
        width: 100%;
    }
    .msg-container.user { justify-content: flex-end; }
    .msg-container.assistant { justify-content: flex-start; }

    /* User bubble (greenish, right aligned) */
    .user-msg {
        background-color:rgb(17, 33, 176); /* */
        color: #000000;
        padding: 10px 14px;
        border-radius: 18px;
        max-width: 70%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    /* Assistant bubble (soft blue, left aligned) */
    .assistant-msg {
        background-color:rgb(92, 2, 4); /* soft light blue */
        color: #000000;
        padding: 10px 14px;
        border-radius: 18px;
        max-width: 70%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "✈️ TripMate ReAct Agent ready. "
                "Ask me to plan, search flights/hotels, check weather, or find sights."
            ),
        }
    ]

# ---- Display Chat History ----
chat_box = st.container()
with chat_box:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        align_class = "user" if role == "user" else "assistant"
        bubble_class = "user-msg" if role == "user" else "assistant-msg"

        st.markdown(
            f'<div class="msg-container {align_class}">'
            f'<div class="{bubble_class}">{content}</div></div>',
            unsafe_allow_html=True,
        )

# ---- User Input ----
if prompt := st.chat_input("Ask me to plan a trip, search flights/hotels, check weather..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show user bubble immediately
    with chat_box:
        st.markdown(
            f'<div class="msg-container user"><div class="user-msg">{prompt}</div></div>',
            unsafe_allow_html=True,
        )

    # Show spinner while agent works
    with st.status("✈️ TripMate is working on your request...", expanded=True) as status:
        try:
            placeholder = st.empty()
            stream_handler = StreamlitCallbackHandler(placeholder)
            result = agent.run({"input": prompt}, callbacks=[stream_handler])
            # reply = result.get("output") or str(result)
            status.update(label="✅ Done!", state="complete")
            if "Final Answer:" not in result:
                reply = f"Final Answer: {result}"
        except Exception as e:
            reply = f"⚠️ Error: {e}"
            status.update(label="❌ Failed", state="error")

    # Stream assistant response
        # Stream assistant response
    streamed_text = ""
    placeholder = st.empty()
    words = reply.split()
    for i in range(len(words)):
        streamed_text += words[i] + " "
        placeholder.markdown(
        f'<div class="msg-container assistant">'
        f'<div class="assistant-msg">{streamed_text}</div></div>',
        unsafe_allow_html=True,
        )
        time.sleep(0.05)  # typing effect


    # Save final assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Refresh so conversation persists
    st.rerun()


