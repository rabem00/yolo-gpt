import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory


class Chatbot:
    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a stand-alone question.
    You can assume that the question is about the information in a CSV file.
    Chat History:
    {chat_history}
    Follow-up entry: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """"You are an AI conversational assistant to answer questions based on information from a csv file.
    You are given data from a csv file and a question, you must help the user find the information they need. 
    Only give responses for information you know about. Don't try to make up an answer.
    Your answers should be short,friendly, in the same language.
    question: {question}
    =========
    {context}
    =======
    """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])  # noqa: E501

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    async def conversational_chat(self, query):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # noqa: E501
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=self.model_name, temperature=self.temperature),
            retriever=self.vectors.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': self.QA_PROMPT},
        )
        result = chain({"question": query, "chat_history": st.session_state["history"]})
        #print(result)
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]
    
    def prompt_form():
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area(
                "Query:",
                placeholder="Ask me anything about the document...",
                key="input",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(label="Send")
            is_ready = submit_button and user_input
        return is_ready, user_input
