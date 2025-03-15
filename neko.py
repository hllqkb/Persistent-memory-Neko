"""
RAG (Retrieval-Augmented Generation) 对话系统实现
====================================

系统架构
--------
本系统实现了一个基于 RAG (检索增强生成) 的对话系统，集成了以下核心组件：

1. 向量数据库 (FAISS)
   - 用于存储对话的向量表示
   - 支持高效的相似度检索
   - 使用 L2 距离进行相似度计算
   - 自动处理数据格式兼容性

2. 图数据库 (Neo4j)
   - 存储结构化的对话历史
   - 支持时间序列查询
   - 使用标签和索引优化查询性能

3. 大语言模型 (Qwen)
   - 通过 SiliconFlow API 访问
   - 支持流式输出
   - 集成上下文增强的对话能力

4. 向量嵌入 (BGE-Large)
   - 使用 BAAI/bge-large-zh-v1.5 模型
   - 1024 维向量表示
   - 通过 SiliconFlow API 获取嵌入

工作流程
--------
1. 用户输入处理
   - 接收用户输入
   - 生成输入文本的向量表示

2. 上下文检索
   - 基于向量相似度搜索相关历史对话
   - 获取最近的对话记录
   - 组合生成增强上下文

3. 响应生成
   - 将上下文注入到系统提示中
   - 调用语言模型生成回答
   - 流式输出响应内容

4. 记忆存储
   - 将对话保存到 Neo4j
   - 生成对话的向量表示
   - 更新 FAISS 索引

关键类说明
--------
1. Memory
   - 对话记忆的数据结构
   - 包含用户消息、AI响应、时间戳和相似度
   - 支持格式化输出

2. Neo4jDatabase
   - 处理与 Neo4j 的交互
   - 管理数据库结构和索引
   - 提供记忆的 CRUD 操作

3. FAISSMemoryStore
   - 管理向量索引和检索
   - 处理数据持久化
   - 提供相似度搜索功能

性能优化
--------
1. 向量检索
   - 使用 FAISS 的 L2 距离索引
   - 支持高效的 K 近邻搜索

2. 数据库优化
   - 时间戳索引加速查询
   - 批量操作减少数据库负载

3. 内存管理
   - 流式处理大型响应
   - 优化数据结构减少内存占用

使用说明
--------
1. 环境要求
   - Python 3.8+
   - Neo4j 数据库
   - FAISS CPU 版本
   - OpenAI API 密钥

2. 配置说明
   - Neo4j 连接参数
   - API 密钥设置
   - 向量维度配置

3. 运行方式
   - 直接对话模式
   - 支持退出命令
   - 自动保存对话历史
"""

from openai import OpenAI
from neo4j import GraphDatabase
import numpy as np
import faiss
import pickle
import os
import json
import datetime
import requests
from typing import List, Dict, Any
import time

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-jivwbgqsesocbzkggntyzjwlkvlyhuiaphesburlvyswzsfc",
    base_url="https://api.siliconflow.cn/v1"
)

class Memory:
    def __init__(self, user_message: str, ai_response: str, timestamp: str, similarity: float = None):
        self.user_message = user_message
        self.ai_response = ai_response
        self.timestamp = timestamp
        self.similarity = similarity

    def __str__(self):
        time_str = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        similarity_str = f" [相似度: {self.similarity:.4f}]" if self.similarity is not None else ""
        return (f"[{time_str}]{similarity_str}\n"
                f"用户: {self.user_message}\n"
                f"助手: {self.ai_response[:100]}..." if len(self.ai_response) > 100 else self.ai_response)

    def short_str(self) -> str:
        """返回简短的记忆描述"""
        time_str = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        similarity_str = f" [相似度: {self.similarity:.4f}]" if self.similarity is not None else ""
        return f"[{time_str}]{similarity_str}\n  问: {self.user_message[:50]}...\n  答: {self.ai_response[:50]}..."

# 初始化 Neo4j
class Neo4jDatabase:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="12345678"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.init_database()

    def init_database(self):
        """初始化数据库结构"""
        with self.driver.session() as session:
            # 创建时间戳索引
            session.run("""
                CREATE INDEX memory_timestamp_idx IF NOT EXISTS
                FOR (m:Memory)
                ON (m.timestamp)
            """)
            # 创建相似度关系索引
            session.run("""
                CREATE INDEX memory_similarity_idx IF NOT EXISTS
                FOR ()-[r:SIMILAR_TO]-()
                ON (r.similarity)
            """)
            print("Neo4j数据库初始化完成")

    def close(self):
        self.driver.close()

    def create_memory_with_relations(self, user_message: str, ai_response: str, similar_memories: List[Memory], similarity_threshold: float = 0.8):
        """创建新的记忆节点并建立相似度关系"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with self.driver.session() as session:
            # 创建新记忆节点
            session.run("""
                CREATE (m:Memory {
                    timestamp: $timestamp,
                    user_message: $user_message,
                    ai_response: $ai_response
                })
            """, timestamp=timestamp,
                user_message=user_message,
                ai_response=ai_response)
            
            # 使用集合去重，只保留相似度最高的关系
            processed_timestamps = set()
            for memory in sorted(similar_memories, key=lambda x: x.similarity, reverse=True):
                if memory.timestamp not in processed_timestamps and memory.similarity >= similarity_threshold:
                    # 使用MATCH和WHERE子句确保节点存在，然后创建关系
                    session.run("""
                        MATCH (m1:Memory {timestamp: $new_timestamp})
                        MATCH (m2:Memory {timestamp: $old_timestamp})
                        WHERE m1 <> m2
                        MERGE (m1)-[r:SIMILAR_TO {similarity: $similarity}]->(m2)
                        MERGE (m2)-[r2:SIMILAR_TO {similarity: $similarity}]->(m1)
                    """,
                        new_timestamp=timestamp,
                        old_timestamp=memory.timestamp,
                        similarity=memory.similarity)
                    processed_timestamps.add(memory.timestamp)
                    print(f"创建关系: 相似度 {memory.similarity:.4f}")
        return timestamp

    def get_related_memories(self, timestamp: str, max_depth: int = 2, min_similarity: float = 0.8) -> List[Memory]:
        """通过图关系获取相关记忆"""
        with self.driver.session() as session:
            # 优化查询语句，使用WHERE过滤掉自身节点
            query = f"""
                MATCH (start:Memory {{timestamp: $timestamp}})
                MATCH path = (start)-[r:SIMILAR_TO*1..{max_depth}]-(related:Memory)
                WHERE start <> related AND
                      ALL(rel IN relationships(path) WHERE rel.similarity >= $min_similarity)
                WITH related, reduce(s = 1.0, rel IN relationships(path) | s * rel.similarity) as total_similarity
                ORDER BY total_similarity DESC
                RETURN DISTINCT related.user_message as user_message,
                       related.ai_response as ai_response,
                       related.timestamp as timestamp,
                       total_similarity
            """
            result = session.run(
                query,
                timestamp=timestamp,
                min_similarity=min_similarity
            )
            
            return [Memory(
                user_message=record["user_message"],
                ai_response=record["ai_response"],
                timestamp=record["timestamp"],
                similarity=record["total_similarity"]
            ) for record in result]

    def get_recent_memories(self, limit: int = 5) -> List[Memory]:
        """获取最近的记忆"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)
                RETURN m.user_message as user_message,
                       m.ai_response as ai_response,
                       m.timestamp as timestamp
                ORDER BY m.timestamp DESC
                LIMIT $limit
            """, limit=limit)
            
            return [Memory(
                user_message=record["user_message"],
                ai_response=record["ai_response"],
                timestamp=record["timestamp"]
            ) for record in result]

# 初始化 FAISS 向量存储
class FAISSMemoryStore:
    def __init__(self, dimension=1024):  # bge-large-zh 模型的维度是1024
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: List[Dict[str, Any]] = []
        self.load_or_create_index()
        print(f"FAISS向量存储初始化完成，维度：{dimension}")

    def load_or_create_index(self):
        if os.path.exists('faiss_index.bin') and os.path.exists('faiss_texts.pkl'):
            self.index = faiss.read_index('faiss_index.bin')
            with open('faiss_texts.pkl', 'rb') as f:
                loaded_texts = pickle.load(f)
                # 处理旧格式数据
                self.texts = []
                for item in loaded_texts:
                    if isinstance(item, str):
                        # 旧格式：直接存储的文本字符串
                        # 使用当前时间作为临时时间戳
                        self.texts.append({
                            "text": item,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        })
                    else:
                        # 新格式：包含text和timestamp的字典
                        self.texts.append(item)
            print(f"已加载现有索引，包含 {len(self.texts)} 条记忆")
            
            # 如果检测到旧格式数据，立即保存为新格式
            if any(isinstance(item, str) for item in loaded_texts):
                print("检测到旧格式数据，正在转换为新格式...")
                self.save_index()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []
            print("创建新的FAISS索引")

    def save_index(self):
        faiss.write_index(self.index, 'faiss_index.bin')
        with open('faiss_texts.pkl', 'wb') as f:
            pickle.dump(self.texts, f)

    def add_text(self, text: str, embedding: np.ndarray, timestamp: str):
        self.texts.append({
            "text": text,
            "timestamp": timestamp
        })
        self.index.add(np.array([embedding]))
        self.save_index()
        print(f"已保存新记忆，当前共有 {len(self.texts)} 条记忆")

    def search(self, query_embedding: np.ndarray, k=3) -> List[Memory]:
        if self.index.ntotal == 0:
            return []
        
        # 确保查询向量的维度正确
        query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"查询向量维度不正确。期望维度: {self.dimension}, 实际维度: {query_embedding.shape[1]}")
        
        # 搜索最相似的k个向量
        distances, indices = self.index.search(query_embedding, k)
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                text = self.texts[idx]["text"]
                timestamp = self.texts[idx]["timestamp"]
                # 将L2距离转换为相似度分数（0-1之间）
                similarity = 1 / (1 + distances[0][i])
                
                # 解析存储的文本
                parts = text.split("\n助手: ")
                if len(parts) == 2:
                    user_message = parts[0].replace("用户: ", "")
                    ai_response = parts[1]
                    
                    memory = Memory(
                        user_message=user_message,
                        ai_response=ai_response,
                        timestamp=timestamp,
                        similarity=similarity
                    )
                    results.append(memory)
        
        return results

    def clear_memory(self):
        """清除所有记忆数据"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        if os.path.exists('faiss_index.bin'):
            os.remove('faiss_index.bin')
        if os.path.exists('faiss_texts.pkl'):
            os.remove('faiss_texts.pkl')
        print("已清除所有FAISS记忆数据")

def get_embedding(text: str) -> np.ndarray:
    """使用 SiliconFlow API 获取文本嵌入向量"""
    start_time = time.time()
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text,
        "encoding_format": "float"
    }
    response = requests.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        embedding = np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
        print(f"获取embedding用时: {time.time() - start_time:.2f}秒")
        return embedding
    else:
        raise Exception(f"获取嵌入向量失败: {response.text}")

# 初始化向量存储和数据库
memory_store = FAISSMemoryStore()
neo4j_db = Neo4jDatabase()

def get_context(query: str, k=3) -> str:
    print(f"\n为查询生成上下文: {query}")
    start_time = time.time()
    
    # 获取查询的embedding
    print("\n1. 生成查询的embedding向量...")
    query_embedding = get_embedding(query)  # 现在返回的是numpy数组
    
    # 从FAISS检索相关内容
    print("\n2. 从向量存储中检索相似记忆...")
    similar_memories = memory_store.search(query_embedding, k)
    if similar_memories:
        print(f"\n找到 {len(similar_memories)} 条相似记忆:")
        print("-" * 50)
        for i, memory in enumerate(similar_memories, 1):
            print(f"记忆 {i}:")
            print(f"时间: {datetime.datetime.strptime(memory.timestamp, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"相似度: {memory.similarity:.4f}")
            print(f"用户问题: {memory.user_message}")
            print(f"AI回答: {memory.ai_response[:100]}..." if len(memory.ai_response) > 100 else memory.ai_response)
            print("-" * 50)
            
            # 3. 从Neo4j获取通过关系连接的相关记忆
            print(f"\n3. 获取与记忆 {i} 相关的记忆链...")
            related_memories = neo4j_db.get_related_memories(memory.timestamp, max_depth=2, min_similarity=0.8)
            if related_memories:
                print(f"\n找到 {len(related_memories)} 条关联记忆:")
                print("-" * 50)
                for j, rel_memory in enumerate(related_memories, 1):
                    print(f"关联记忆 {j}:")
                    print(f"时间: {datetime.datetime.strptime(rel_memory.timestamp, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"关联强度: {rel_memory.similarity:.4f}")
                    print(f"用户问题: {rel_memory.user_message}")
                    print(f"AI回答: {rel_memory.ai_response[:100]}..." if len(rel_memory.ai_response) > 100 else rel_memory.ai_response)
                    print("-" * 50)
    else:
        print("没有找到相似的历史记忆")
    
    # 组合上下文
    print("\n4. 组合上下文信息...")
    context = "以下是相关的历史对话记录：\n"
    
    # 添加相似对话及其关联记忆
    if similar_memories:
        context += "\n相似的历史对话及其关联记忆：\n"
        for memory in similar_memories:
            context += f"\n[主要相似记忆] {str(memory)}\n"
            related_memories = neo4j_db.get_related_memories(memory.timestamp, max_depth=2, min_similarity=0.8)
            if related_memories:
                context += "\n关联记忆：\n"
                for rel_memory in related_memories:
                    context += f"[关联强度: {rel_memory.similarity:.4f}] {str(rel_memory)}\n"
            context += "-" * 50 + "\n"
    
    print(f"\n生成上下文完成，总用时: {time.time() - start_time:.2f}秒")
    return context

def save_conversation(user_message: str, ai_response: str):
    print("\n保存对话...")
    start_time = time.time()
    
    # 获取相似记忆
    combined_text = f"用户: {user_message}\n助手: {ai_response}"
    embedding = get_embedding(combined_text)  # 现在返回的是numpy数组
    similar_memories = memory_store.search(embedding, k=5)
    
    # 保存到Neo4j并建立关系
    print("\n1. 保存到Neo4j并建立记忆关系...")
    timestamp = neo4j_db.create_memory_with_relations(user_message, ai_response, similar_memories)
    
    # 保存到FAISS
    print("\n2. 保存到FAISS向量存储...")
    memory_store.add_text(combined_text, embedding, timestamp)
    
    print(f"\n保存对话完成，总用时: {time.time() - start_time:.2f}秒")

try:
    print("\n=== AI助手启动 ===")
    
    while True:
        message = input("\n请输入你想说的话（输入'exit'退出）：\n")
        if message.lower() == "exit":
            break
            
        # 获取相关上下文
        context = get_context(message)
        
        # 构建带有上下文的提示
        print("\n构建带有上下文的提示...")
        messages = [
            {"role": "system", "content": "你是一个有记忆能力的AI助手。" + context},
            {"role": "user", "content": message},
        ]

        # 获取AI响应
        print("\n生成回答中...")
        start_time = time.time()
        response = client.chat.completions.create(
            model="ft:LoRA/Qwen/Qwen2.5-32B-Instruct:lqyic99t1y:s:jkbyjnqzspvrgwcbuktk",# 32b微调的角色扮演模型，需要结合prompt使用
            messages=messages,
            stream=True,
            max_tokens=4096
        )

        # 收集完整的响应
        print("\nAI助手回答：")
        print("-" * 50)
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            print(content, end='', flush=True)
        print("\n" + "-" * 50)
        
        print(f"\n生成回答用时: {time.time() - start_time:.2f}秒")
            
        # 保存对话
        save_conversation(message, full_response)

finally:
    print("\n=== 清理资源 ===")
    neo4j_db.close()
    print("Neo4j连接已关闭")
    print("程序结束")