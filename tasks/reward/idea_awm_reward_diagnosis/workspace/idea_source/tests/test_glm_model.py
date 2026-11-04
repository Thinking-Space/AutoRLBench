"""Optional hosted API example; importing it does not make an API call."""

import os


def main():
    api_key = os.environ.get("ZAI_API_KEY", "")
    if not api_key or api_key == "EMPTY":
        raise SystemExit("Set your own ZAI_API_KEY before running this optional example")
    from zai import ZhipuAiClient

    client = ZhipuAiClient(api_key=api_key)
    response = client.chat.completions.create(
        model=os.environ.get("ZAI_MODEL", "glm-5.3-flash"),
        messages=[{
            "content": [{
                "type": "text",
                "text": "Where is the second bottle of beer from the right on the table?  Provide coordinates in [[xmin,ymin,xmax,ymax]] format",
            }],
            "role": "user",
        }],
        thinking={"type": "enabled"},
    )
    print(response.choices[0].message)


if __name__ == "__main__":
    main()
