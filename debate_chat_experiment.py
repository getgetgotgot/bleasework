"""Run: python -m streamlit run debate_chat_experiment.py

Keep debate2write_arg_gen.py and its prompts/ folder beside this file.
Set HF_TOKEN in your environment or Streamlit secrets. No model downloads.
"""

import importlib.util
import json
import os
import re
from pathlib import Path

import streamlit as st
from huggingface_hub import InferenceClient


def setting(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def complete(client, model, messages, max_tokens):
    response = client.chat_completion(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=0.7, top_p=0.9,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The provider returned an empty answer. Please try again.")
    return content.strip()


def generate_argument(client, model, proposition, max_tokens, progress, reply_words=250):
    # A fresh module isolates this session's client without changing the source file
    # or the globals of a module shared with another Streamlit session.
    source = Path(__file__).resolve().with_name("debate2write_arg_gen.py")
    spec = importlib.util.spec_from_file_location("debate_experiment_backend", source)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    count = 0
    stages = ["Creating personas", "Selecting personas", "Modeling the debate",
              "Drafting the plan", "Writing the argument"]

    def generate(prompt, model=model, max_tokens=max_tokens):
        nonlocal count
        progress.write(stages[count] if count < len(stages) else "Continuing generation")
        count += 1
        return complete(client, model, [{"role": "user", "content": prompt}], max_tokens)

    backend.openai_generate = generate
    generator = backend.ArgumentGenerator(model=model, max_length=max_tokens)
    generator.surfacegen_prompt += (
        f"\nKeep the final article concise: aim for {reply_words} words and do not "
        f"exceed {reply_words} words. Return only the article."
    )
    result = generator.generate_one(proposition)
    if not result or not result.get("argument"):
        raise RuntimeError("No argument was generated. Try again or start a new debate.")
    return result


def friendly_error(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status == 402:
        return "Your Hugging Face account needs inference credits. Add credits, then retry."
    if status in (401, 403):
        return "Check your HF_TOKEN and its Make calls to Inference Providers permission."
    if status == 429:
        return "The provider is busy or rate-limited. Wait a moment, then retry."
    if isinstance(error, FileNotFoundError):
        return "A required file is missing. Keep the original script and prompts folder beside this app."
    if isinstance(error, AssertionError):
        return "The model did not return at least three readable personas. See details below."
    if status == 400:
        return "The provider rejected the request parameters. See details below."
    if status in (404, 422) or isinstance(error, StopIteration):
        return "The selected model/provider may be unavailable or incompatible. See details below."
    if status and status >= 500:
        return "The inference provider encountered a server error. Please retry later."
    return "Generation failed. Open Error details below to see the cause."


def error_details(error, token=""):
    detail = str(error) or "No error message was provided."
    if token:
        detail = detail.replace(token, "[REDACTED]")
    detail = re.sub(r"hf_[A-Za-z0-9_-]+", "[REDACTED]", detail)
    detail = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", detail)
    return f"{type(error).__name__}: {detail[:4000]}"


def main():
    st.set_page_config(page_title="Debate chat experiment", page_icon="💬")
    st.title("Debate chat")
    st.caption("Enter a proposition to generate a counterargument, then discuss or revise it.")
    if "debate_messages" not in st.session_state:
        st.session_state.debate_messages = []
        st.session_state.debate_result = None

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Hosted model", value=setting("HF_MODEL_ID", "openai/gpt-oss-120b"))
        provider = st.text_input("Provider", value=setting("HF_PROVIDER", "auto"))
        reply_words = st.number_input("Reply length (words)", min_value=50, max_value=1000,
                                     value=250, step=50)
        st.caption("A word target for the final argument and chat replies; exact counts may vary.")
        max_tokens = st.slider("Maximum output tokens per call", 1000, 8000, 4000, 500)
        st.caption("An initial argument uses at least five API calls. Each follow-up uses one. Inference credits apply.")
        st.caption("Chat history lasts for this browser session. Download it before refreshing or starting over.")
        if st.button("New debate"):
            st.session_state.debate_messages = []
            st.session_state.debate_result = None
            st.rerun()

    token = setting("HF_TOKEN")
    if not token:
        st.info("Set HF_TOKEN in your environment or Streamlit secrets, then restart the app.")

    result = st.session_state.debate_result
    if result:
        with st.expander("Personas, discussion, and plan"):
            st.json(result["persona_lists"])
            st.markdown("### Discussion")
            st.write(result["discussion"])
            st.markdown("### Plan")
            st.write(result["plan"])
        transcript = "\n\n".join(
            f"{message['role'].capitalize()}:\n{message['content']}"
            for message in st.session_state.debate_messages
        )
        st.download_button("Download conversation (.txt)", transcript,
                           file_name="debate_chat.txt", mime="text/plain")
        st.download_button("Download full result (.json)",
                           json.dumps({"initial_result": result,
                                       "messages": st.session_state.debate_messages},
                                      ensure_ascii=False, indent=2),
                           file_name="debate_chat.json", mime="application/json")

    for message in st.session_state.debate_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Enter a proposition" if not result else "Ask a question or request a revision",
        disabled=not bool(token),
    )
    if not prompt:
        return
    if not model.strip() or not provider.strip():
        st.error("Enter a model and provider in Settings.")
        return

    user_message = {"role": "user", "content": prompt}
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        client = InferenceClient(api_key=token, provider=provider.strip(), timeout=180)
        new_result = None
        with st.chat_message("assistant"):
            if result is None:
                with st.status("Generating your counterargument…", expanded=True) as progress:
                    new_result = generate_argument(client, model.strip(), prompt, max_tokens, progress, reply_words)
                    progress.update(label="Counterargument ready", state="complete", expanded=False)
                answer = new_result["argument"]
            else:
                # Keep the original argument plus the last six exchanges to limit input size.
                history = st.session_state.debate_messages
                context = history if len(history) <= 14 else history[:2] + history[-12:]
                messages = [{"role": "system", "content":
                    "You help discuss and revise an argumentative article. The original "
                    "article refutes the user's proposition. Follow revision requests, "
                    "distinguish factual claims from opinions, and never invent sources. "
                    f"Keep each reply concise: aim for {reply_words} words and do not "
                    f"exceed {reply_words} words. Shorter replies are fine when sufficient."}]
                with st.spinner("Thinking…"):
                    answer = complete(client, model.strip(), messages + context + [user_message], max_tokens)
            st.markdown(answer)
        if new_result is not None:
            st.session_state.debate_result = new_result
        st.session_state.debate_messages.extend([user_message, {"role": "assistant", "content": answer}])
        st.rerun()
    except Exception as error:
        st.error(friendly_error(error))
        with st.expander("Error details", expanded=True):
            st.code(error_details(error, token), language=None)
            st.write(f"Model: {model.strip()} | Provider: {provider.strip()}")
        st.caption("Your previous conversation is kept. Submit your message again to retry.")


if __name__ == "__main__":
    main()
