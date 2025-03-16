import os
import logging
from logging.handlers import RotatingFileHandler
from openai import OpenAI
from neo4j import GraphDatabase
import numpy as np
import faiss
import pickle
import json
import datetime
import requests
from typing import List, Dict, Any
import time
from colorama import Fore, Back, Style, init

# 初始化colorama
init(autoreset=True)  # autoreset=True 使每次打印后自动恢复默认颜色

# 在 Memory 类之前添加日志配置
def setup_logger():
    """配置日志记录器"""
    logger = logging.getLogger('neko')
    logger.setLevel(logging.INFO)
    
    # 创建 logs 目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 设置日志文件，使用 RotatingFileHandler 进行日志轮转
    handler = RotatingFileHandler(
        'logs/neko.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()

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

    def create_memory_with_relations(self, user_message: str, ai_response: str, similar_memories: List[Memory], similarity_threshold: float = 0.7):#Neo4j关系连接度0.7
        """创建新的记忆节点并建立相似度关系
        
        Args:
            user_message: 用户消息
            ai_response: AI回答
            similar_memories: 相似记忆列表
            similarity_threshold: 相似度阈值，低于此值的记忆不建立关系
        """
        print("Neo4j关系连接度",similarity_threshold)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with self.driver.session() as session:
            # 首先检查是否存在高度相似的记忆
            for memory in similar_memories:
                if memory.similarity > 0.95:  # 相似度超过95%视为重复
                    print(f"发现高度相似的记忆 (相似度: {memory.similarity:.4f})，跳过创建")
                    return memory.timestamp  # 直接返回已存在的记忆时间戳
            
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
            
            # 按相似度降序排序
            sorted_memories = sorted(similar_memories, key=lambda x: x.similarity, reverse=True)
            
            for memory in sorted_memories:
                if memory.timestamp not in processed_timestamps and similarity_threshold <= memory.similarity <= 0.95:
                    # 检查是否已经存在相似的关系路径
                    existing_relations = session.run("""
                        MATCH path = (m1:Memory)-[r:SIMILAR_TO*]-(m2:Memory)
                        WHERE m1.timestamp = $timestamp1 AND m2.timestamp = $timestamp2
                        RETURN count(path) as path_count
                    """, timestamp1=timestamp, timestamp2=memory.timestamp).single()
                    
                    if existing_relations and existing_relations["path_count"] > 0:
                        print(f"已存在关系路径，跳过创建 (相似度: {memory.similarity:.4f})")
                        continue
                    
                    # 创建新的关系
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
        """通过图关系获取相关记忆
        
        Args:
            timestamp: 起始记忆的时间戳
            max_depth: 最大搜索深度
            min_similarity: 最小相似度阈值 (0.0-1.0)
            
        Returns:
            List[Memory]: 相关记忆列表，按相似度降序排序，去重
        """
        with self.driver.session() as session:
            # 优化查询语句:
            # 1. 使用 DISTINCT 去重
            # 2. 添加相似度过滤
            # 3. 优化路径搜索
            query = f"""
                MATCH (start:Memory {{timestamp: $timestamp}})
                MATCH path = (start)-[r:SIMILAR_TO*1..{max_depth}]-(related:Memory)
                WHERE start <> related AND
                      ALL(rel IN relationships(path) WHERE rel.similarity >= $min_similarity)
                WITH DISTINCT related,
                     reduce(s = 1.0, rel IN relationships(path) | s * rel.similarity) as total_similarity,
                     length(path) as path_length
                ORDER BY total_similarity DESC, path_length ASC
                RETURN related.user_message as user_message,
                       related.ai_response as ai_response,
                       related.timestamp as timestamp,
                       total_similarity
            """
            result = session.run(
                query,
                timestamp=timestamp,
                min_similarity=min_similarity
            )
            
            # 使用集合去重
            seen_timestamps = set()
            memories = []
            
            for record in result:
                timestamp = record["timestamp"]
                if timestamp not in seen_timestamps:
                    memories.append(Memory(
                        user_message=record["user_message"],
                        ai_response=record["ai_response"],
                        timestamp=timestamp,
                        similarity=record["total_similarity"]
                    ))
                    seen_timestamps.add(timestamp)
            
            return memories

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
        return embedding
    else:
        raise Exception(f"获取嵌入向量失败: {response.text}")

# 初始化向量存储和数据库
memory_store = FAISSMemoryStore()
neo4j_db = Neo4jDatabase()

def get_context(query: str, k=3) -> str:
    print(f"\n{Fore.CYAN}为查询生成上下文: {Fore.YELLOW}{query}{Style.RESET_ALL}")
    start_time = time.time()
    
    # 获取查询的embedding
    print(f"\n{Fore.CYAN}1. 生成查询的embedding向量...{Style.RESET_ALL}")
    query_embedding = get_embedding(query)
    
    # 从FAISS检索相关内容
    print(f"\n{Fore.CYAN}2. 从向量存储中检索相似记忆...{Style.RESET_ALL}")
    similar_memories = memory_store.search(query_embedding, k)
    if similar_memories:
        print(f"\n{Fore.GREEN}找到 {len(similar_memories)} 条相似记忆:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")
        for i, memory in enumerate(similar_memories, 1):
            print(f"{Fore.YELLOW}记忆 {i}:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}相似度: {memory.similarity:.4f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}用户问题: {memory.user_message}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}AI回答: {memory.ai_response[:100]}...{Style.RESET_ALL}" if len(memory.ai_response) > 100 else f"{Fore.BLUE}AI回答: {memory.ai_response}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")
            
            # 3. 从Neo4j获取通过关系连接的相关记忆
            print(f"\n{Fore.CYAN}3. 获取与记忆 {i} 相关的记忆链...{Style.RESET_ALL}")
            related_memories = neo4j_db.get_related_memories(memory.timestamp, max_depth=2, min_similarity=0.8)
            if related_memories:
                print(f"\n{Fore.GREEN}找到 {len(related_memories)} 条关联记忆:{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")
                for j, rel_memory in enumerate(related_memories, 1):
                    print(f"{Fore.MAGENTA}关联记忆 {j}:{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}关联强度: {rel_memory.similarity:.4f}{Style.RESET_ALL}")
                    print(f"{Fore.WHITE}用户问题: {rel_memory.user_message}{Style.RESET_ALL}")
                    print(f"{Fore.BLUE}AI回答: {rel_memory.ai_response[:100]}...{Style.RESET_ALL}" if len(rel_memory.ai_response) > 100 else f"{Fore.BLUE}AI回答: {rel_memory.ai_response}{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}没有找到相似的历史记忆{Style.RESET_ALL}")
    
    # 组合上下文
    print(f"\n{Fore.CYAN}4. 组合上下文信息...{Style.RESET_ALL}")
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
    return context

def save_conversation(user_message: str, ai_response: str):
    """保存对话并记录日志"""
    print(f"\n{Fore.CYAN}保存对话...{Style.RESET_ALL}")
    start_time = time.time()
    
    # 获取相似记忆
    combined_text = f"用户: {user_message}\n助手: {ai_response}"
    embedding = get_embedding(combined_text)
    similar_memories = memory_store.search(embedding, k=5)
    
    # 检查是否存在高度相似的记忆
    is_duplicate = False
    if similar_memories:
        highest_similarity = similar_memories[0].similarity
        if highest_similarity > 0.95:  # 相似度阈值
            is_duplicate = True
            logger.info(f"检测到重复问题，相似度: {highest_similarity:.4f}")
            logger.info(f"原问题: {similar_memories[0].user_message}")
            logger.info(f"新问题: {user_message}")
            print(f"{Fore.RED}检测到重复问题，跳过保存（相似度: {highest_similarity:.4f}）{Style.RESET_ALL}")
            return
    
    # 保存到Neo4j并建立关系
    print(f"\n{Fore.CYAN}1. 保存到Neo4j并建立记忆关系...{Style.RESET_ALL}")
    timestamp = neo4j_db.create_memory_with_relations(user_message, ai_response, similar_memories)
    
    # 保存到FAISS
    print(f"\n{Fore.CYAN}2. 保存到FAISS向量存储...{Style.RESET_ALL}")
    memory_store.add_text(combined_text, embedding, timestamp)
    
    # 计算费用并正确解包元组
    input_tokens, output_tokens, cost = calculate_tokens_and_cost(user_message, ai_response)
    
    # 记录日志
    logger.info("\n============ 新对话已保存 ============\n")
    logger.info(f"时间戳: {timestamp}")
    logger.info(f"用户问题: {user_message}")
    logger.info(f"AI回答: {ai_response[:200]}..." if len(ai_response) > 200 else ai_response)
    logger.info(f"费用: ￥{cost:.7f}")
    
    if similar_memories:
        logger.info("相似记忆:")
        for i, memory in enumerate(similar_memories[:3], 1):  # 只记录前3条相似记忆
            logger.info(f"  {i}. 相似度: {memory.similarity:.4f}")
            logger.info(f"     问题: {memory.user_message[:100]}..." if len(memory.user_message) > 100 else memory.user_message)

def calculate_tokens_and_cost(input_text: str, output_text: str) -> tuple:
    """计算输入输出的token数和总费用
    
    Returns:
        tuple: (input_tokens: int, output_tokens: int, total_cost: float)
    """
    # 简单估算：中文每个字约1.5个token，英文每个单词约1个token
    # 这是粗略估计，实际token数可能有所不同
    input_tokens = len(input_text) * 1.5
    output_tokens = len(output_text) * 1.5
    
    # 计算费用 (单位：元)
    # 输入：￥4/M tokens = ￥0.000004/token
    # 输出：￥16/M tokens = ￥0.000016/token
    input_cost = input_tokens * 0.000004
    output_cost = output_tokens * 0.000016
    total_cost = input_cost + output_cost
    
    return int(input_tokens), int(output_tokens), total_cost

try:
    print(f"\n{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 15}{Fore.WHITE} AI助手启动 {Fore.YELLOW}{'=' * 15}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}欢迎使用持久记忆AI助手！输入'exit'可退出系统。{Style.RESET_ALL}")
    logger.info("\n================ AI助手启动 ================\n")
    total_cost = 0.0
    
    while True:
        print(f"\n{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        message = input(f"{Fore.CYAN}【用户输入】{Fore.WHITE} 请输入您的问题（输入'exit'退出）：\n{Fore.YELLOW}>>> {Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        if message.lower() == "exit":
            break
            
        # 记录用户输入
        logger.info("\n================ 新对话开始 ================\n")
        logger.info(f"用户输入: {message}")
        
        # 获取相关上下文
        context = get_context(message)
        
        # 构建带有上下文的提示
        print(f"\n{Fore.CYAN}构建带有上下文的提示...{Style.RESET_ALL}")
        system_message = "你是一个有记忆能力的AI助手。" + context
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
        
        # 记录完整prompt
        logger.info("完整Prompt:")
        logger.info(f"System: {system_message}")
        logger.info(f"User: {message}")

        # 获取AI响应
        print(f"\n{Fore.CYAN}生成回答中...{Style.RESET_ALL}")
        start_time = time.time()
        response = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3",
            messages=messages,
            stream=True,
            max_tokens=4096
        )

        # 收集完整的响应
        print(f"\n{Fore.GREEN}{'*' * 50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'*' * 15}{Fore.WHITE} AI助手回答 {Fore.GREEN}{'*' * 15}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'*' * 50}{Style.RESET_ALL}")
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            print(f"{Fore.WHITE}{content}{Style.RESET_ALL}", end='', flush=True)
        print(f"\n{Fore.GREEN}{'*' * 50}{Style.RESET_ALL}")
        
        # 计算token数和费用
        input_tokens, output_tokens, cost = calculate_tokens_and_cost(
            system_message + message, 
            full_response
        )
        total_cost += cost

        
        # 保存对话
        save_conversation(message, full_response)
                
        # 打印token统计和费用信息
        print(f"\n{Fore.YELLOW}{'#' * 50}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'#' * 15}{Fore.WHITE} 对话统计 {Fore.YELLOW}{'#' * 15}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'#' * 50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}输入tokens: {Fore.WHITE}{input_tokens:,} {Fore.YELLOW}(约￥{input_tokens * 0.000004:.4f}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}输出tokens: {Fore.WHITE}{output_tokens:,} {Fore.YELLOW}(约￥{output_tokens * 0.000016:.4f}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}本次费用: {Fore.GREEN}￥{cost:.4f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}累计总费用: {Fore.GREEN}￥{total_cost:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'#' * 50}{Style.RESET_ALL}")
        
        # 记录token统计和费用信息
        logger.info(f"输入tokens: {input_tokens:,}")
        logger.info(f"输出tokens: {output_tokens:,}")
        logger.info(f"本次费用: ￥{cost:.4f}")
        logger.info(f"累计总费用: ￥{total_cost:.4f}")

finally:
    logger.info(f"================ 会话结束 ================")
    logger.info(f"本次会话总费用: ￥{total_cost:.4f}")
    neo4j_db.close()
    print(f"\n{Fore.RED}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 15}{Fore.WHITE} 系统关闭中 {Fore.YELLOW}{'=' * 15}{Style.RESET_ALL}")
    print(f"{Fore.RED}{'=' * 50}{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}本次会话总费用: ￥{total_cost:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Neo4j连接已关闭{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}感谢使用持久记忆AI助手，再见！{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}程序结束{Style.RESET_ALL}")