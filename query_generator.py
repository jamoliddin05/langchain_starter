import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()


class QueryHandler:
    def __init__(self, vectorstore_path: str, openai_api_key: str):
        # Load OpenAI API key and initialize the embeddings model
        self.token = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.token)

        # Load the vectorstore from the specified path
        self.vectorstore = FAISS.load_local(
            vectorstore_path, self.embeddings, allow_dangerous_deserialization=True
        )

    def get_query(self, query: str) -> tuple[str, list]:
        """
        Returns a formatted query with instructions, context, and the original question.

        :param query: The question/query from the user.
        :return: Formatted string with context and the query.
        """
        custom_query = """
            Begin by carefully reading the given pieces of retrieved context, paying close attention to the details provided to answer the question.
            Analyze the context to identify the key information necessary to answer the question effectively. Look for keywords and relevant details that can guide your response.
            Once you have identified the essential information, extrapolate the details needed to formulate a clear and concise answer to the question presented.
            Organize your response in a logical and coherent manner, starting with a brief summary of the context followed by a direct answer to the question.
            Include all necessary information from the retrieved context to support your answer and ensure it aligns with the given pieces of information.
            Use clear and unambiguous language in your response to avoid any confusion or misinterpretation. Be as straightforward as possible to provide a precise and accurate answer.
            Take into account any limitations or uncertainties in the context provided and address them in your response. If additional context is needed to answer the question effectively, consider asking follow-up questions to gather more information.
            If there are any ambiguities in the context or question, seek clarifications before proceeding with your response to ensure accuracy.
            Provide a well-structured and detailed response that comprehensively addresses the question using the available context.
            Double-check your response to ensure it accurately answers the question based on the given context and information. Verify that your answer aligns with the retrieved pieces of context provided.

            NOTE: Do not include any "ANSWER:" or "Response:" headers, just return the answer text directly. Do not use any formatting, just return the plain text.

            CONTEXT:
            {context}

            QUESTION:
            {question}
        """

        # Perform similarity search on the vectorstore
        results = self.vectorstore.similarity_search(query, k=2)
        context_list = [res.page_content for res in results]

        # Create the context string with bullet points
        context = "".join([f"\n\t\t\t. {res.page_content}" for res in results])

        # Fill the custom_query with the dynamic context and question
        filled_query = custom_query.format(context=context, question=f"\t{query}")

        return filled_query, context_list
