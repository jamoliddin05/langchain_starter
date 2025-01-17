from langchain_openai import OpenAI
from query_generator import QueryHandler
import os

if __name__ == '__main__':
    token = os.getenv('OPENAI_API_KEY')
    vectorstore_path = os.getenv('VECTORSTORE_PATH')

    query_handler = QueryHandler(vectorstore_path, token)
    llm = OpenAI(api_key=token)

    # Start the CLI interface with a while True loop
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")

        if 'quit' in query.lower():
            print("Exiting...")
            break

        formatted_query, context_list = query_handler.get_query(query)

        response = llm.invoke(formatted_query)

        print("\n--- CONTEXT ---")
        for context in context_list:
            print(f"{context}")

        # Print the response from OpenAI
        print("\n--- RESPONSE ---")
        print(f"{response.strip()}")