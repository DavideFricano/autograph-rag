from autograph_rag.augmentation.prompter import PromptGenerator

gen = PromptGenerator()


def test_system_prompt_contains_text():
    result = gen.build_system_prompt("Sei un medico.")
    assert "SISTEMA:" in result
    assert "Sei un medico." in result

def test_query_prompt_contains_text():
    result = gen.build_query_prompt("Qual è la diagnosi?")
    assert "DOMANDA:" in result
    assert "Qual è la diagnosi?" in result

def test_context_prompt_contains_text():
    result = gen.build_context_prompt("Il paziente ha febbre.")
    assert "CONTESTO:" in result
    assert "Il paziente ha febbre." in result

def test_context_prompt_empty_shows_no_source():
    assert "Nessuna fonte trovata" in gen.build_context_prompt("")

def test_full_prompt_preserves_all_content():
    result = gen.build_prompt("Sistema.", "Domanda?", "Contesto.")
    assert "Sistema." in result
    assert "Domanda?" in result
    assert "Contesto." in result
