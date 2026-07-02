"""
═══════════════════════════════════════════════════════════════════
示例 18：对话式 RAG（Conversational RAG）—— RAG + 聊天记忆
═══════════════════════════════════════════════════════════════════
目的：把 RAG（示例04）和 Memory（示例06）结合起来，实现"能连续追问的
知识库问答"。这是最常见的企业应用模式——客服机器人、内部知识库助手。

核心概念：
  - 示例04 的问题：每次提问都是独立的，不能追问
  - 示例06 的问题：只有对话记忆，没有知识库
  - 本示例：两者结合 → "根据文档回答，且能在上一轮基础上追问"

流程：
  用户问"什么是RAG？"  → 检索文档 + 回答
  用户追问"和普通搜索有什么区别？" → LLM 结合上文理解"它"指 RAG + 检索

运行方式：设置了 DEEPSEEK_API_KEY 即可
═══════════════════════════════════════════════════════════════════
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ────────────────────────────────────────
# 第 1 步：准备知识库（同示例04）
# ────────────────────────────────────────
documents = [
    "LangChain 是一个用于构建 LLM 应用的开源框架，支持 Python 和 JS。",
    "LangChain 的核心概念包括：Chain（链）、Agent（智能体）、Tool（工具）、Memory（记忆）。",
    "RAG 是 Retrieval-Augmented Generation 的缩写，意思是检索增强生成。",
    "RAG 的核心优势是：让 LLM 能回答训练数据之外的私有知识，且答案可溯源。",
    "和普通搜索的区别：搜索返回网页链接列表，RAG 直接生成基于资料的准确回答。",
    "LangSmith 是 LangChain 旗下的 LLM 应用调试与监控平台，可以追踪每次调用。",
    "LCEL 全称 LangChain Expression Language，用管道符 | 串联组件。",
    "向量数据库是一种专门存储和检索向量的数据库，常用有 Chroma、FAISS、Pinecone。",
]

# 切分 + 向量化
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = text_splitter.create_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="deepseek-chat", temperature=0,
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ────────────────────────────────────────
# 第 2 步：关键！把历史问题"压缩"成一个检索查询
# ────────────────────────────────────────
# 用户追问时说"它"、"这个"——这些词拿去检索是查不到的
# 需要先让 LLM 结合聊天历史，把追问"翻译"成一个独立的检索语句

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """根据聊天历史，把用户的最新问题改写成一个"独立的、不依赖上下文就能理解的"检索语句。
如果最新问题已经足够独立，直接原样返回。只返回改写后的问题，不要解释。"""),
    MessagesPlaceholder("history"),  # ← 之前的对话
    ("human", "{input}"),            # ← 用户最新说的话
])

# 这个链负责：结合上文 → 输出一个独立的检索语句
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

# ────────────────────────────────────────
# 第 3 步：手动实现 "历史感知的检索器"
# ────────────────────────────────────────
# 流程：原始问题 → 结合历史改写 → 用改写后的问题检索

class HistoryAwareRetriever:
    """包装检索器：先改写问题，再检索"""
    def __init__(self, retriever, contextualize_chain, history):
        self.retriever = retriever
        self.contextualize_chain = contextualize_chain
        self.history = history

    def invoke(self, question: str):
        # 1. 结合历史改写问题
        rewritten = self.contextualize_chain.invoke({
            "history": self.history,
            "input": question,
        })
        # 2. 用改写后的问题检索
        return self.retriever.invoke(rewritten)


# ────────────────────────────────────────
# 第 4 步：RAG 生成链（含历史上下文）
# ────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """根据以下资料回答问题。如果资料中没有相关信息，请如实说明不知道。

资料：
{context}"""),
    MessagesPlaceholder("history"),   # ← 对话历史也传给 LLM
    ("human", "{question}"),
])

rag_chain = rag_prompt | llm | StrOutputParser()

# ────────────────────────────────────────
# 第 5 步：多轮对话演示
# ────────────────────────────────────────
history = []  # 维护对话历史

def chat(question: str):
    """一轮对话：改写问题 → 检索 → 结合历史生成回答"""
    # 创建"历史感知"检索器
    aware_retriever = HistoryAwareRetriever(retriever, contextualize_chain, history)

    # 检索
    docs = aware_retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    # 生成
    answer = rag_chain.invoke({
        "context": context,
        "history": history,
        "question": question,
    })

    # 更新历史
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))

    return answer, docs


print("【对话式 RAG —— 多轮追问演示】")
print("知识库内容：LangChain / RAG 相关文档\n")
print("─" * 60)

# 第1轮：正常提问
answer, docs = chat("什么是 RAG？")
print(f"🤔 第1轮：什么是 RAG？")
print(f"🤖 {answer}\n")

# 第2轮：用代词追问（关键测试！）
answer, docs = chat("它和普通搜索有什么区别？")
print(f"🤔 第2轮：它和普通搜索有什么区别？")
print(f"🤖 {answer}\n")
print(f"   （注意：LLM 理解'它'指的是 RAG，而不是瞎猜）")

# 第3轮：更深入的追问
answer, docs = chat("那它的核心优势是什么？")
print(f"\n🤔 第3轮：那它的核心优势是什么？")
print(f"🤖 {answer}")

print("\n─" * 60)
print("💡 对比示例04的普通RAG：如果问'它是什么？'，普通RAG没法理解'它'指什么")
