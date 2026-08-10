"""RAG 第三步：验证真实 Embedding API 调用。"""

import sys

if __package__:
    from app.embedding_client import EmbeddingAPIError, EmbeddingConfig, embed_texts
else:
    from embedding_client import EmbeddingAPIError, EmbeddingConfig, embed_texts


def main() -> None:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = input("请输入要向量化的一句话：").strip()

    try:
        vectors = embed_texts([text], EmbeddingConfig.from_environment())
    except (EmbeddingAPIError, ValueError) as error:
        print(f"调用失败：{error}")
        return

    vector = vectors[0]
    print("调用成功。")
    print(f"文本：{text}")
    print(f"向量维度：{len(vector)}")
    print(f"向量前 8 个数字：{vector[:8]}")


if __name__ == "__main__":
    main()
