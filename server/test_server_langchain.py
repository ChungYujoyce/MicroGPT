from langchain_community.llms import VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://172.18.0.2:5000/v1",
    model_name="test",
    max_tokens=128,
    temperature=0,
    model_kwargs={
        "stop": ["."],
    },
)
print(llm.invoke("<s> [INST] Hello! How are you? [/INST]"))