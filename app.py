import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor

# --- StreamlitのSecretsからAPIキー取得 ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
pinecone_index_name = st.secrets.get("PINECONE_INDEX_NAME")

# Streamlitのページ設定
st.set_page_config(page_title="Note RAG", page_icon="📝")

# --- モデルの初期化 (キャッシュして高速化) ---
@st.cache_resource
def load_index():
    #モデルの設定
    Settings.llm = Gemini(model="gemini-2.5-flash-lite", temperature=0.5)

    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")

    Settings.embed_model = embed_model

    # Pineconeへの接続
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)
    # 既存のインデックスを参照
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    #データの読み込みとインデックスの作成
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

def check_password():
    #パスワードが正しいかチェック
    def password_entered():
        #入力されたパスワードを確認するコールバック
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # パスワードを状態から消去して安全に
        else:
            st.session_state["password_correct"] = False

    # すでに認証済みならTrueを返す
    if st.session_state.get("password_correct", False):
        return True

    # ログイン画面を表示
    st.title("🔒 認証が必要です")
    st.text_input(
        "パスワードを入力してください", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state:
        st.error("😕 パスワードが違います")
    return False

# --- メイン処理 ---
if check_password():

    #アプリのメイン処理
    try:
        index = load_index()
        # チャットエンジンの作成
        if "chat_engine" not in st.session_state:
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="condense_question", 
                verbose=True,
                similarity_top_k=5, #関連する上位n記事
                node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.80) 
            ]
            )

        st.title("📝 新潟市店舗記事チャットボット")

        # チャット履歴の表示
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ユーザー入力
        if prompt := st.chat_input("質問を入力してください"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("検索中..."):
                    response = st.session_state.chat_engine.chat(prompt)
                    st.markdown(response.response)

                    # === 追加: 参照元の表示 ===
                    # ソースノードからメタデータを抽出
                    sources = []
                    seen_urls = set() # 重複排除用
                    
                    for node in response.source_nodes:
                        # メタデータの取得（ingest時に保存した title と url）
                        metadata = node.metadata
                        url = metadata.get("url", "#")
                        title = metadata.get("title", "無題のドキュメント")
                        
                        # URLが重複していない場合のみリストに追加
                        if url not in seen_urls and url != "#":
                            sources.append(f"- [{title}]({url})")
                            seen_urls.add(url)
                    
                    # 参照元があれば表示
                    if sources:
                        st.markdown("---")
                        st.markdown("### 📚 参照元")
                        st.markdown("\n".join(sources))
                    # ==========================
            
            st.session_state.messages.append({"role": "assistant", "content": response.response})

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")