
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use ONLY the Retrieved Policy Information. Do not use outside knowledge or assume what hotels "usually" do.
    2. Start with a direct answer to the question in the first sentence, then add the supporting details.
    3. If the question names a property (for example ASH-CHI Loop), use only the figures listed for that property. Never transfer a fee, time, or rule from one property to another.
    4. Quote exact values as written: times (with "local property time"), dollar amounts, per night vs. per stay, caps, pet limits, and "plus applicable tax" where stated.
    5. Include the conditions, exceptions, and approvals that change the answer (for example: subject to availability, manager approval, rate-plan or confirmation terms, evidence requirements).
    6. If the question has several parts, answer every part.
    7. Do not promise refunds, waivers, upgrades, or exceptions. State who may approve them and what must be verified first.
    8. If the retrieved information does not cover the property, amenity, or situation asked about, say: "The available policy documents do not contain enough information to answer this question." Then state which part cannot be confirmed and advise checking the live property profile or contacting the property team. Do not guess.
    9. If the retrieved information conflicts, mention the conflict clearly.
    10. Keep the answer concise (2 to 6 sentences or short bullets) and leave out information unrelated to the question. Do not mention "chunks", "context", metadata, or file paths.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
