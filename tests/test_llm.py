"""Tests for src.llm — run with: python tests/test_llm.py"""

from src.llm import chat, achat, count_tokens


def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0
    print("  count_tokens OK")


def test_chat():
    resp = chat("Reply with exactly: OK", model="deepseek/deepseek-chat")
    assert isinstance(resp, str) and len(resp) > 0
    print(f"  chat: {resp!r}")


def test_chat_with_system():
    resp = chat(
        "hello",
        model="deepseek/deepseek-chat",
        system="Reply with exactly one word: bonjour",
    )
    assert isinstance(resp, str) and len(resp) > 0
    print(f"  chat+system: {resp!r}")


if __name__ == "__main__":
    print("test_count_tokens...")
    test_count_tokens()

    print("test_chat...")
    test_chat()

    print("test_chat_with_system...")
    test_chat_with_system()

    print("\nAll tests passed.")
