import os
import feedparser
import trafilatura
import hashlib
from pinecone import Pinecone
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# =================設定=================
# 対象noteのRSSのURL
RSS_URL = "https://note.com/niigata_omise/rss" 
# ======================================

def get_article_text(url):
    """URLから本文を抽出する"""
    downloaded = trafilatura.fetch_url(url)
    return trafilatura.extract(downloaded)

def generate_id(url):
    """URLから一意のIDを生成する（MD5ハッシュ）"""
    return hashlib.md5(url.encode()).hexdigest()

def update():
    print("🔄 RSSフィードを確認中...")
    feed = feedparser.parse(RSS_URL)
    
    if not feed.entries:
        print("⚠ 記事が見つかりませんでした。URLを確認してください。")
        return

    # Pinecone接続
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    
    # 新しい記事リスト
    new_docs = []
    
    print(f"🔍 最新 {len(feed.entries)} 件の記事をチェックします...")

    # 既存のIDをチェックするためにFetchする（効率化のため）
    for entry in feed.entries:
        url = entry.link
        doc_id = generate_id(url)
        
        # PineconeにIDが存在するか確認
        fetch_response = pinecone_index.fetch(ids=[doc_id])
        
        if not fetch_response.vectors:
            print(f"🆕 新規記事発見: {entry.title}")
            text = get_article_text(url)
            if text:
                # メタデータ付きでドキュメント化
                doc = Document(
                    text=text,
                    id_=doc_id,
                    metadata={
                        "title": entry.title,
                        "url": url,
                        "published": entry.published
                    }
                )
                new_docs.append(doc)
            else:
                print(f"⚠ 本文抽出失敗: {url}")
        else:
            print(f"✅ 登録済み: {entry.title}")

    if not new_docs:
        print("🎉 新しい記事はありませんでした。")
        return

    # 新規記事がある場合のみ、重いモデルをロードする
    print(f"🚀 {len(new_docs)} 件の記事をベクトル化します（モデルロード中...）")
    
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large"
    )
    Settings.embed_model = embed_model
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # インデックスに追加（Upsert）
    VectorStoreIndex.from_documents(
        new_docs,
        storage_context=storage_context,
        show_progress=True
    )
    print("✨ 更新完了しました！")

if __name__ == "__main__":
    # ローカルテスト用（環境変数がなければエラーになります）
    if "PINECONE_API_KEY" not in os.environ:
        print("❌ PINECONE_API_KEY が設定されていません。")
    else:
        update()