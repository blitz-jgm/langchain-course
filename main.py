import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from langchain-course!")
    print(f"Your API key is: {os.environ.get('ANTHROPIC_API_KEY')}")


if __name__ == "__main__":
    main()
