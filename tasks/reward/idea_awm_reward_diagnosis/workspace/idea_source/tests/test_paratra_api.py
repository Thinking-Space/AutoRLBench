"""Optional hosted-service smoke test; provide your own service settings."""

import os

import openai


def main():
    required = ("AWM_JUDGE_API_KEY", "AWM_JUDGE_BASE_URL", "AWM_JUDGE_MODEL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("Set your own service settings: " + ", ".join(missing))
    client = openai.OpenAI(
        api_key=os.environ["AWM_JUDGE_API_KEY"],
        base_url=os.environ["AWM_JUDGE_BASE_URL"],
    )
    response = client.chat.completions.create(
        model=os.environ["AWM_JUDGE_MODEL"],
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=50,
    )
    print(response)


if __name__ == "__main__":
    main()
