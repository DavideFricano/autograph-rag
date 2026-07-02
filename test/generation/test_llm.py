from autograph_rag.augmentation.prompter import PromptGenerator
from autograph_rag.generation.llm import OllamaClient, OpenAIClient


def test_ollama_payload_contains_model():
    client = OllamaClient(model="llama3")
    payload = client._payload("prompt", temperature=0.1, num_ctx=4096, stream=False)
    assert payload["model"] == "llama3"

def test_ollama_payload_contains_prompt():
    client = OllamaClient(model="llama3")
    payload = client._payload("Testo del prompt.", temperature=0.1, num_ctx=4096, stream=False)
    assert payload["prompt"] == "Testo del prompt."

def test_ollama_payload_stream_flag():
    client = OllamaClient(model="llama3")
    assert client._payload("p", 0.1, 512, stream=True)["stream"] is True
    assert client._payload("p", 0.1, 512, stream=False)["stream"] is False

def test_openai_messages_roles():
    client = OpenAIClient.__new__(OpenAIClient)
    client.prompt_generator = PromptGenerator()
    messages = client._messages("sistema", "domanda", "contesto")
    assert [m["role"] for m in messages] == ["system", "user"]

def test_openai_messages_contain_content():
    client = OpenAIClient.__new__(OpenAIClient)
    client.prompt_generator = PromptGenerator()
    messages = client._messages("Sei un medico.", "Qual è la diagnosi?", "Il paziente ha febbre.")
    assert "Sei un medico." in messages[0]["content"]
    assert "Qual è la diagnosi?" in messages[1]["content"]
    assert "Il paziente ha febbre." in messages[1]["content"]
