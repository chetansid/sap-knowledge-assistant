from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def load_qa_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
    You are an SAP expert assistant. Answer the question based ONLY on the 
    context provided below from SAP documentation. If the answer is not in 
    the context, say "I could not find this in the provided documentation."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(question: str) -> dict:
    chain, retriever = load_qa_chain()
    
    # Get answer
    answer = chain.invoke(question)
    
    # Get source pages separately
    source_docs = retriever.invoke(question)
    sources = []
    for doc in source_docs:
        page = doc.metadata.get("page", "unknown")
        if page != "unknown" and page not in sources:
            sources.append(page + 1)

    return {
        "question": question,
        "answer": answer,
        "source_pages": sorted(sources)
    }


if __name__ == "__main__":
    print("SAP Knowledge Assistant ready. Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ")
        if question.lower() == "quit":
            break
        result = ask_question(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Source pages: {result['source_pages']}\n")