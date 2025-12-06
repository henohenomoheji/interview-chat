import streamlit as st
from src.utils import *
import os
from dotenv import load_dotenv

from langchain_openai import (
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    ChatOpenAI
)
from langchain_core.messages import (
    HumanMessage, 
    AIMessage,
)

USER_NAME = "user"
ASSISTANT_NAME = "assistant"
load_dotenv('/workspaces/interview-chat/.env')



def main():
    st.title("Interview Chat")

        # emmbeddingsのモデルを取得
    embeddings = None
    # if os.getenv('AZURE_OPENAI_API_KEY') != "":
    #     embeddings = AzureOpenAIEmbeddings(
    #         azure_deployment="embedding",
    #         openai_api_version="2024-06-01"
    #     )
    if os.getenv('OPENAI_API_KEY') != "":
        # OpenAIの場合
        embeddings = OpenAIEmbeddings()
    else:
        st.error("APIKeyの設定を確認してください")

    # chatのモデルを取得
    model = None
    # Azureの場合
    # if os.getenv('AZURE_OPENAI_API_KEY') != "":
    #     model = AzureChatOpenAI(
    #         azure_deployment="chat",
    #         openai_api_version="2024-06-01",
    #     )
    if os.getenv('OPENAI_API_KEY') != "":
        # OpenAIの場合
        model = ChatOpenAI(model="gpt-4")
    else:
        st.error("APIKeyの設定を確認してください")

        # Chain取得
    contextualize_chain = get_contextualize_prompt_chain(model)
    chain = get_chain(model)

        # FAISSからretrieverを取得
    retriever = pull_from_faiss(embeddings)

    if 'chat_log' not in st.session_state:
        st.session_state.chat_log = []

    # st.session_state.chat_log = [
    #     {"name": USER_NAME,"msg":"こんにちは"},
    #     {"name": ASSISTANR_NAME,"msg":"こんばんは"},
    # ]


    user_msg = st.chat_input("Input")

    if user_msg:

        # 以前のチャットログを表示
        for chat in st.session_state.chat_log:
            if isinstance(chat, AIMessage):
                with st.chat_message(ASSISTANT_NAME):
                    st.write(chat.content)
            else:
                with st.chat_message(USER_NAME):
                    st.write(chat.content)



        with st.chat_message(USER_NAME):
            st.write(user_msg)

        # 質問を修正する
        if st.session_state.chat_log:
            new_msg = contextualize_chain.invoke({"chat_history": st.session_state.chat_log, "input": user_msg})
        else:
            new_msg = user_msg
        print(user_msg, "=>", new_msg)

        # 類似ドキュメントを取得
        relavant_docs = retriever.invoke(new_msg, k=3)


        response = ""
        with st.chat_message(ASSISTANT_NAME):
            msg_placeholder = st.empty()
            
            for r in chain.stream({"chat_history": st.session_state.chat_log, "context": relavant_docs, "input": user_msg}):
                response += r.content
                msg_placeholder.markdown(response + "■")
            msg_placeholder.markdown(response)

        # セッションにチャットログを追加
        st.session_state.chat_log.extend([
            HumanMessage(content=user_msg),
            AIMessage(content=response)
        ])
    

if __name__ == "__main__":
    main()
