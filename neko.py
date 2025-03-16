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
from typing import List, Dict, Any, Optional
import time
from colorama import Fore, Back, Style, init
import jieba.analyse
import re
import yaml

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
    
    # 设置日志文件，使用 FileHandler 而不是 RotatingFileHandler
    # 'w' 模式会覆盖已存在的日志文件
    handler = logging.FileHandler(
        'logs/neko.log',
        mode='w',  # 使用 'w' 模式覆盖已存在的日志文件
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # 移除所有已存在的处理器
    logger.handlers.clear()
    
    # 添加新的处理器
    logger.addHandler(handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()

# 将 load_config 函数定义移到文件前面
def load_config():
    """加载配置文件"""
    try:
        # 首先尝试加载 YAML 格式
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config:
                    print(f"{Fore.GREEN}YAML配置文件加载成功{Style.RESET_ALL}")
                    return config
        except Exception as yaml_error:
            print(f"{Fore.YELLOW}加载YAML配置失败: {str(yaml_error)}, 尝试加载JSON配置{Style.RESET_ALL}")
        
        # 尝试加载标准 JSON
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"{Fore.GREEN}JSON配置文件加载成功{Style.RESET_ALL}")
                return config
        except Exception as json_error:
            print(f"{Fore.YELLOW}加载JSON配置失败: {str(json_error)}{Style.RESET_ALL}")
            raise  # 重新抛出异常，触发默认配置
            
    except Exception as e:
        print(f"{Fore.RED}加载配置文件失败: {str(e)}{Style.RESET_ALL}")
        logger.error(f"加载配置文件失败: {str(e)}")
        # 返回默认配置
        return {
            "api": {
                "key": "sk-jivwbgqsesocbzkggntyzjwlkvlyhuiaphesburlvyswzsfc",
                "base_url": "https://api.siliconflow.cn/v1",
                "timeout": 30
            },
            "model": {
                "name": "Pro/deepseek-ai/DeepSeek-V3",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "embedding": {
                "model": "BAAI/bge-large-zh-v1.5",
                "timeout": 30
            },
            "storage": {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "12345678"
                },
                "faiss": {
                    "dimension": 1024,
                    "index_type": "flat"
                }
            }
        }

# 初始化日志记录器
logger = setup_logger()

# 加载配置
config = load_config()

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=config["api"]["key"],
    base_url=config["api"]["base_url"]
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
        # 确保密码是字符串类型
        if not isinstance(password, str):
            password = str(password)
        
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
            # 创建主题索引
            session.run("""
                CREATE INDEX memory_topic_idx IF NOT EXISTS
                FOR (m:Memory)
                ON (m.topic)
            """)
            print("Neo4j数据库初始化完成")

    def close(self):
        self.driver.close()

    def create_memory_with_relations(self, user_message: str, ai_response: str, similar_memories: List[Memory], similarity_threshold: float = 0.7):
        """创建新的记忆节点并建立相似度关系
        
        Args:
            user_message: 用户消息
            ai_response: AI回答
            similar_memories: 相似记忆列表
            similarity_threshold: 相似度阈值，低于此值的记忆不建立关系
        """
        print("Neo4j关系连接度", similarity_threshold)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # 提取主题
        topic = extract_topic(user_message)
        
        # 创建预览
        user_message_preview = user_message[:100] + "..." if len(user_message) > 100 else user_message
        ai_response_preview = ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
        
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
                    user_message_preview: $user_message_preview,
                    ai_response_preview: $ai_response_preview,
                    topic: $topic,
                    created_at: datetime()
                })
            """, timestamp=timestamp,
                user_message_preview=user_message_preview,
                ai_response_preview=ai_response_preview,
                topic=topic
            )
            
            # 使用集合去重，只保留相似度最高的关系
            processed_timestamps = set()
            
            # 按相似度降序排序
            sorted_memories = sorted(similar_memories, key=lambda x: x.similarity, reverse=True)
            
            for memory in sorted_memories:
                if memory.timestamp not in processed_timestamps and similarity_threshold <= memory.similarity <= 0.95:
                    # 检查是否已经存在相似的关系路径，限制深度为10
                    existing_relations = session.run("""
                        MATCH path = (m1:Memory)-[r:SIMILAR_TO*1..10]-(m2:Memory)
                        WHERE m1.timestamp = $timestamp1 AND m2.timestamp = $timestamp2
                        AND length(path) <= 10
                        RETURN count(path) as path_count
                    """, timestamp1=timestamp, timestamp2=memory.timestamp).single()
                    
                    if existing_relations and existing_relations["path_count"] > 0:
                        print(f"已存在关系路径，跳过创建 (相似度: {memory.similarity:.4f})")
                        continue
                    
                    # 创建新的关系，同时检查路径深度
                    session.run("""
                        MATCH (m1:Memory {timestamp: $new_timestamp})
                        MATCH (m2:Memory {timestamp: $old_timestamp})
                        WHERE m1 <> m2
                        AND NOT EXISTS((m1)-[:SIMILAR_TO*1..10]-(m2))
                        MERGE (m1)-[r:SIMILAR_TO {similarity: $similarity}]->(m2)
                        MERGE (m2)-[r2:SIMILAR_TO {similarity: $similarity}]->(m1)
                    """,
                        new_timestamp=timestamp,
                        old_timestamp=memory.timestamp,
                        similarity=memory.similarity)
                    processed_timestamps.add(memory.timestamp)
                    print(f"创建关系: 相似度 {memory.similarity:.4f}")
        
        return timestamp

    def get_related_memories(self, timestamp: str, max_depth: int = 10, min_similarity: float = 0.75) -> List[Memory]:
        """通过图关系获取相关记忆
        
        Args:
            timestamp: 起始记忆的时间戳
            max_depth: 最大搜索深度，默认为10
            min_similarity: 最小相似度阈值 (0.0-1.0)
            
        Returns:
            List[Memory]: 相关记忆列表，按相似度降序排序，去重
        """
        with self.driver.session() as session:
            query = f"""
                MATCH (start:Memory {{timestamp: $timestamp}})
                MATCH path = (start)-[r:SIMILAR_TO*1..{max_depth}]-(related:Memory)
                WHERE start <> related 
                AND ALL(rel IN relationships(path) WHERE rel.similarity >= $min_similarity)
                AND length(path) <= {max_depth}
                WITH DISTINCT related,
                     reduce(s = 1.0, rel IN relationships(path) | s * rel.similarity) as total_similarity,
                     length(path) as path_length
                ORDER BY total_similarity DESC, path_length ASC
                RETURN related.timestamp as timestamp,
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
                ts = record["timestamp"]
                if ts not in seen_timestamps:
                    # 从FAISS获取完整记忆
                    memory = memory_store.get_memory_by_timestamp(ts)
                    
                    # 如果FAISS中没有，则从Neo4j获取预览
                    if not memory:
                        memory = self.get_memory_by_timestamp(ts)
                    
                    if memory:
                        memory.similarity = record["total_similarity"]
                        memories.append(memory)
                        seen_timestamps.add(ts)
        
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

    def get_recent_related_memories(self, query_embedding: np.ndarray, limit: int = 3) -> List[Memory]:
        """获取最近的相关记忆，结合时间和相关度
        
        Args:
            query_embedding: 查询的向量嵌入
            limit: 返回的记忆数量
            
        Returns:
            List[Memory]: 相关记忆列表
        """
        # 首先获取最近的记忆
        recent_memories = self.get_recent_memories(limit=10)  # 获取更多记忆，后续会筛选
        
        if not recent_memories:
            return []
            
        # 计算这些记忆与查询的相似度
        for memory in recent_memories:
            # 构建完整文本用于计算相似度
            memory_text = f"用户: {memory.user_message}\n助手: {memory.ai_response}"
            memory_embedding = get_embedding(memory_text)
            
            # 计算余弦相似度
            similarity = np.dot(query_embedding, memory_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
            )
            memory.similarity = float(similarity)
        
        # 按相似度排序
        recent_memories.sort(key=lambda x: x.similarity, reverse=True)
        
        # 获取相似度最高的几条记忆
        top_memories = recent_memories[:limit]
        
        # 对于每条记忆，获取其相关记忆链
        result_memories = []
        for memory in top_memories:
            result_memories.append(memory)
            
            # 获取相关记忆
            related_memories = self.get_related_memories(
                memory.timestamp, 
                max_depth=1,  # 减小深度以提高效率
                min_similarity=0.7  # 降低阈值以获取更多相关记忆
            )
            
            # 只添加前2条相关记忆，避免结果过多
            for related in related_memories[:2]:
                if related not in result_memories:
                    result_memories.append(related)
        
        # 最终限制总数量
        return result_memories[:limit]

    def get_related_memories_by_topic(self, topic: str, limit: int = 5) -> List[str]:
        """根据主题获取相关记忆的时间戳
        
        Args:
            topic: 主题关键词
            limit: 返回的记忆数量上限
            
        Returns:
            List[str]: 相关记忆的时间戳列表
        """
        with self.driver.session() as session:
            # 首先检查是否有带 topic 属性的节点
            check_result = session.run("""
                MATCH (m:Memory)
                WHERE m.topic IS NOT NULL
                RETURN count(m) as count
            """).single()
            
            if check_result and check_result["count"] > 0:
                # 如果有带 topic 属性的节点，使用 topic 进行查询
                result = session.run("""
                    MATCH (m:Memory)
                    WHERE m.topic CONTAINS $topic
                    RETURN m.timestamp as timestamp
                    ORDER BY m.timestamp DESC
                    LIMIT $limit
                """, topic=topic, limit=limit)
                
                return [record["timestamp"] for record in result]
            else:
                # 如果没有带 topic 属性的节点，使用用户消息预览进行模糊匹配
                result = session.run("""
                    MATCH (m:Memory)
                    WHERE m.user_message_preview CONTAINS $topic
                    RETURN m.timestamp as timestamp
                    ORDER BY m.timestamp DESC
                    LIMIT $limit
                """, topic=topic, limit=limit)
                
                return [record["timestamp"] for record in result]

    def get_memory_by_timestamp(self, timestamp: str) -> Optional[Memory]:
        """根据时间戳从Neo4j获取记忆，如果Neo4j中没有完整内容，则从FAISS获取
        
        Args:
            timestamp: 记忆的时间戳
            
        Returns:
            Optional[Memory]: 找到的记忆对象，未找到则返回None
        """
        with self.driver.session() as session:
            # 首先从Neo4j获取记忆预览
            result = session.run("""
                MATCH (m:Memory {timestamp: $timestamp})
                RETURN m.user_message_preview as user_preview, 
                       m.ai_response_preview as ai_preview,
                       m.topic as topic
            """, timestamp=timestamp).single()
            
            if not result:
                return None
                
            # 从FAISS获取完整内容
            full_memory = memory_store.get_memory_by_timestamp(timestamp)
            
            if full_memory:
                # 如果FAISS中有完整内容，直接返回
                return full_memory
            else:
                # 如果FAISS中没有完整内容，使用Neo4j中的预览创建Memory对象
                return Memory(
                    user_message=result["user_preview"],
                    ai_response=result["ai_preview"],
                    timestamp=timestamp
                )

def export_memories(format="markdown", keyword=None):
    """导出记忆为可读格式
    
    Args:
        format: 导出格式，支持 "markdown", "json", "txt"
        keyword: 可选的过滤关键词
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_export_{timestamp}"
        if keyword:
            filename = f"memory_export_{keyword}_{timestamp}"
        
        # 从Neo4j获取记忆
        with neo4j_db.driver.session() as session:
            query = """
                MATCH (m:Memory)
                {}
                RETURN m.timestamp as timestamp, m.topic as topic, 
                       m.user_message_preview as user_msg, m.ai_response_preview as ai_msg
                ORDER BY m.timestamp DESC
            """
            
            if keyword:
                filter_clause = """
                    WHERE m.user_message_preview CONTAINS $keyword OR 
                          m.ai_response_preview CONTAINS $keyword OR
                          m.topic CONTAINS $keyword
                """
                query = query.format(filter_clause)
                result = session.run(query, keyword=keyword)
                print(f"{Fore.CYAN}使用关键词过滤: {Fore.YELLOW}{keyword}{Style.RESET_ALL}")
            else:
                query = query.format("")
                result = session.run(query)
            
            memories = [(
                record["timestamp"], 
                record["topic"], 
                record["user_msg"], 
                record["ai_msg"]
            ) for record in result]
        
        if not memories:
            print(f"{Fore.YELLOW}没有找到记忆，无法导出{Style.RESET_ALL}")
            return False
        
        # 根据格式导出
        if format.lower() == "markdown":
            filename = f"{filename}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("# 记忆导出\n\n")
                f.write(f"导出时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for ts, topic, user_msg, ai_msg in memories:
                    f.write(f"## {topic} ({ts})\n\n")
                    f.write(f"**用户**: {user_msg}\n\n")
                    f.write(f"**助手**: {ai_msg}\n\n")
                    f.write("---\n\n")
        
        elif format.lower() == "json":
            filename = f"{filename}.json"
            export_data = []
            for ts, topic, user_msg, ai_msg in memories:
                export_data.append({
                    "timestamp": ts,
                    "topic": topic,
                    "user_message": user_msg,
                    "ai_response": ai_msg
                })
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        elif format.lower() == "txt":
            filename = f"{filename}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"记忆导出 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for ts, topic, user_msg, ai_msg in memories:
                    f.write(f"[{ts}] {topic}\n")
                    f.write(f"用户: {user_msg}\n")
                    f.write(f"助手: {ai_msg}\n")
                    f.write("-" * 50 + "\n\n")
        
        else:
            print(f"{Fore.RED}不支持的导出格式: {format}{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.GREEN}记忆已导出为: {filename}{Style.RESET_ALL}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}导出记忆时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"导出记忆时发生错误: {str(e)}")
        return False

print(f"{Fore.GREEN}export:{Style.RESET_ALL} - 导出所有记忆为Markdown格式")
print(f"{Fore.GREEN}export:格式{Style.RESET_ALL} - 导出所有记忆为指定格式 (markdown/json/txt)")
print(f"{Fore.GREEN}export:关键词{Style.RESET_ALL} - 导出包含特定关键词的记忆")
print(f"{Fore.GREEN}export:格式 关键词{Style.RESET_ALL} - 导出包含特定关键词的记忆为指定格式")
# 初始化 FAISS 向量存储
class FAISSMemoryStore:
    def __init__(self, dimension=1024, index_type="flat", index_path="faiss_index.pkl"):
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = index_path
        self.texts = []
        
        # 尝试加载现有索引
        if os.path.exists(self.index_path):
            try:
                print(f"{Fore.GREEN}加载FAISS索引文件: {self.index_path}{Style.RESET_ALL}")
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.index = data['index']
                    self.texts = data.get('texts', [])
                print(f"{Fore.GREEN}FAISS索引加载成功，包含 {len(self.texts)} 条记忆{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}加载FAISS索引失败: {str(e)}，将创建新索引{Style.RESET_ALL}")
                logger.error(f"加载FAISS索引失败: {str(e)}")
                self._create_new_index()
        else:
            print(f"{Fore.YELLOW}FAISS索引文件不存在，创建新索引{Style.RESET_ALL}")
            self._create_new_index()
            # 立即保存空索引，避免下次启动时再次提示
            self.save_index()
            print(f"{Fore.GREEN}已创建并保存新的FAISS索引{Style.RESET_ALL}")
    
    def _create_new_index(self):
        """创建新的FAISS索引"""
        if self.index_type.lower() == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type.lower() == "ivf":
            # IVF索引需要训练，这里使用简单的随机数据
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            # 生成随机训练数据
            np.random.seed(42)
            train_data = np.random.random((1000, self.dimension)).astype('float32')
            self.index.train(train_data)
        else:
            # 默认使用Flat索引
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.texts = []

    def save_index(self):
        """保存索引到文件"""
        try:
            # 确保目录存在
            index_dir = os.path.dirname(self.index_path)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir)
                
            with open(self.index_path, 'wb') as f:
                pickle.dump({'index': self.index, 'texts': self.texts}, f)
            print(f"{Fore.GREEN}FAISS索引已保存，包含 {len(self.texts)} 条记忆{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}保存FAISS索引失败: {str(e)}{Style.RESET_ALL}")
            logger.error(f"保存FAISS索引失败: {str(e)}")
            return False

    def add_text(self, text: str, embedding: np.ndarray, timestamp: str):
        """添加新的记忆到FAISS索引
        
        Args:
            text: 完整对话文本 (格式: "用户: xxx\n助手: xxx")
            embedding: 文本的向量表示
            timestamp: 时间戳，作为唯一标识
        """
        # 确保存储完整对话内容
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
        if os.path.exists('faiss_index.pkl'):
            os.remove('faiss_index.pkl')
        if os.path.exists('faiss_index.bin'):
            os.remove('faiss_index.bin')
        if os.path.exists('faiss_texts.pkl'):
            os.remove('faiss_texts.pkl')
        print("已清除所有FAISS记忆数据")

    def get_memory_by_timestamp(self, timestamp: str) -> Optional[Memory]:
        """根据时间戳获取完整记忆
        
        Args:
            timestamp: 记忆的时间戳
            
        Returns:
            Optional[Memory]: 找到的记忆对象，未找到则返回None
        """
        for item in self.texts:
            if item.get("timestamp") == timestamp:
                text = item["text"]
                parts = text.split("\n助手: ")
                if len(parts) == 2:
                    user_message = parts[0].replace("用户: ", "")
                    ai_response = parts[1]
                    return Memory(
                        user_message=user_message,
                        ai_response=ai_response,
                        timestamp=timestamp
                    )
        return None

def get_embedding(text: str) -> np.ndarray:
    """使用 SiliconFlow API 获取文本嵌入向量"""
    if not text or not isinstance(text, str):
        raise ValueError("输入文本不能为空且必须是字符串类型")
        
    # 清理和预处理文本
    text = text.strip()
    if not text:
        raise ValueError("输入文本不能全为空白字符")
    
    # 准备API请求
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json"
    }
    
    # 从配置中获取embedding模型
    embedding_model = config["embedding"]["model"]
    timeout = config["embedding"]["timeout"]
    
    # API请求数据
    data = {
        "model": embedding_model,
        "input": text,
        "encoding_format": "float"
    }
    
    try:
        # 发送请求
        response = requests.post(
            f"{config['api']['base_url']}/embeddings",
            headers=headers,
            json=data,
            timeout=timeout
        )
        
        # 检查响应状态
        if response.status_code != 200:
            error_msg = f"API请求失败 (状态码: {response.status_code}): {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # 解析响应
        result = response.json()
        
        # 检查响应格式
        if not isinstance(result, dict) or 'data' not in result:
            raise Exception(f"API返回格式错误: {result}")
            
        if not result['data'] or not isinstance(result['data'], list):
            raise Exception(f"API返回数据为空或格式错误: {result}")
            
        # 获取embedding
        embedding = result['data'][0]['embedding']
        
        # 转换为numpy数组
        return np.array(embedding, dtype=np.float32)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求失败: {str(e)}")
        raise Exception(f"获取embedding失败: {str(e)}")
        
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"处理API响应时出错: {str(e)}")
        raise Exception(f"处理embedding响应失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"获取embedding时发生未知错误: {str(e)}")
        if response and response.text:
            logger.error(f"API响应: {response.text}")
        raise Exception(f"获取embedding失败: {str(e)}")

def extract_topic(text: str) -> str:
    """从文本中提取主题关键词
    
    简单实现：提取最重要的2-3个名词或动词
    更复杂的实现可以使用NLP库如jieba进行关键词提取
    
    Args:
        text: 输入文本
        
    Returns:
        str: 提取的主题关键词，以空格分隔
    """
    # 提取前3个关键词
    keywords = jieba.analyse.extract_tags(text, topK=3)
    return " ".join(keywords)

# 初始化向量存储和数据库
memory_store = FAISSMemoryStore()
neo4j_db = Neo4jDatabase(
    uri=config["storage"]["neo4j"]["uri"],
    user=config["storage"]["neo4j"]["user"],
    password=config["storage"]["neo4j"]["password"]
)

# 从base.md文件获取basemd变量的内容
try:
    with open('base.md', 'r', encoding='utf-8') as file:
        basemd = file.read()
except FileNotFoundError:
    logger.error("base.md文件不存在")
    basemd = "base.md文件不存在，无法获取内容。"

def get_context(query: str, k=3) -> str:
    """获取相关上下文，返回简洁的prompt给AI，同时记录详细日志"""
    logger.info(f"\n============ 开始构建上下文 ============")
    logger.info(f"查询: {query}")
    print(f"\n{Fore.CYAN}为查询生成上下文: {Fore.YELLOW}{query}{Style.RESET_ALL}")
    
    # 获取查询的embedding和主题
    print(f"\n{Fore.CYAN}1. 生成查询向量...{Style.RESET_ALL}")
    query_embedding = get_embedding(query)
    topic = extract_topic(query)
    
    logger.info(f"提取的主题关键词: {topic}")
    print(f"\n{Fore.CYAN}提取的主题关键词: {Fore.YELLOW}{topic}{Style.RESET_ALL}")
    
    # 1. 从FAISS获取向量相似度高的记忆
    print(f"\n{Fore.CYAN}2. 从FAISS获取向量相似度高的记忆...{Style.RESET_ALL}")
    vector_memories = memory_store.search(query_embedding, k=k*2)  # 获取更多候选
    
    # 2. 从Neo4j获取主题相关的记忆时间戳
    print(f"\n{Fore.CYAN}3. 从Neo4j获取主题相关的记忆...{Style.RESET_ALL}")
    topic_timestamps = neo4j_db.get_related_memories_by_topic(topic, limit=k*2)
    topic_memories = []
    
    # 获取主题相关记忆的完整内容
    for ts in topic_timestamps:
        memory = memory_store.get_memory_by_timestamp(ts)
        if memory:
            # 计算与查询的相似度
            memory_text = f"{memory.user_message}\n{memory.ai_response}"
            memory_embedding = get_embedding(memory_text)
            similarity = np.dot(query_embedding, memory_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
            )
            memory.similarity = float(similarity)
            topic_memories.append(memory)
    
    # 3. 从Neo4j获取图关系相关的记忆
    print(f"\n{Fore.CYAN}4. 从Neo4j获取图关系相关的记忆...{Style.RESET_ALL}")
    # 如果有向量相似的记忆，使用其中最相似的作为起点
    graph_memories = []
    if vector_memories:
        start_timestamp = vector_memories[0].timestamp
        graph_memories = neo4j_db.get_related_memories(
            start_timestamp, 
            max_depth=config["retrieval"]["graph_related_depth"],
            min_similarity=config["retrieval"]["min_similarity"]
        )
    
    # 4. 合并所有记忆并去重
    print(f"\n{Fore.CYAN}5. 合并记忆并去重...{Style.RESET_ALL}")
    all_memories = vector_memories + topic_memories + graph_memories
    
    # 使用字典去重，保留相似度最高的版本
    unique_memories = {}
    for memory in all_memories:
        if memory.timestamp not in unique_memories or memory.similarity > unique_memories[memory.timestamp].similarity:
            unique_memories[memory.timestamp] = memory
    
    # 转换回列表并按相似度排序
    merged_memories = list(unique_memories.values())
    merged_memories.sort(key=lambda x: x.similarity, reverse=True)
    
    # 5. 进一步筛选，移除内容高度相似的记忆
    print(f"\n{Fore.CYAN}6. 筛选内容高度相似的记忆...{Style.RESET_ALL}")
    filtered_memories = []
    seen_contents = set()
    
    for memory in merged_memories:
        # 创建内容指纹 (简化版本，实际可以使用更复杂的算法)
        content_key = f"{memory.user_message[:100]}"
        
        # 检查是否有高度相似的内容已经被选中
        is_duplicate = False
        for existing_key in seen_contents:
            # 使用简单的字符串相似度检查
            if similarity_score(content_key, existing_key) > config["retrieval"]["filter_similarity_threshold"]:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_memories.append(memory)
            seen_contents.add(content_key)
            
            # 如果已经有足够的记忆，停止添加
            if len(filtered_memories) >= k:
                break
    
    # 记录详细日志
    logger.info(f"找到相关记忆: 向量相似 {len(vector_memories)}条, 主题相关 {len(topic_memories)}条, 图关系 {len(graph_memories)}条")
    logger.info(f"合并后: {len(merged_memories)}条, 过滤后: {len(filtered_memories)}条")
    
    for i, memory in enumerate(filtered_memories, 1):
        logger.info(f"\n记忆 {i}:")
        logger.info(f"相似度: {memory.similarity:.4f}")
        logger.info(f"时间戳: {memory.timestamp}")
        logger.info(f"用户: {memory.user_message}")
        logger.info(f"助手: {memory.ai_response}")
    
    # 打印简要统计
    print(f"\n{Fore.GREEN}检索结果: {len(filtered_memories)} 条记忆 (从 {len(merged_memories)} 条候选中筛选){Style.RESET_ALL}")
    
    # 构建简洁的prompt
    prompt = ""
    if filtered_memories:
        for memory in filtered_memories:
            prompt += f"用户: {memory.user_message}\n"
            prompt += f"助手: {memory.ai_response}\n\n"
    
    logger.info("\n============ 上下文构建完成 ============\n")
    return prompt

# 辅助函数：计算两个字符串的相似度
def similarity_score(str1, str2):
    """计算两个字符串的简单相似度 (0-1)"""
    # 使用集合计算Jaccard相似度
    set1 = set(str1)
    set2 = set(str2)
    
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union

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
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # 保存到Neo4j并建立关系
    print(f"\n{Fore.CYAN}1. 保存到Neo4j并建立记忆关系...{Style.RESET_ALL}")
    neo4j_timestamp = neo4j_db.create_memory_with_relations(user_message, ai_response, similar_memories)
    
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

# 添加数据迁移函数，用于更新现有的Neo4j节点

def update_existing_nodes_with_topic():
    """更新现有节点，添加主题属性"""
    try:
        with neo4j_db.driver.session() as session:
            # 检查是否有节点缺少topic属性
            result = session.run("""
                MATCH (m:Memory)
                WHERE m.topic IS NULL AND m.user_message_preview IS NOT NULL
                RETURN count(m) as count
            """).single()
            
            missing_topic_count = result["count"] if result else 0
            
            if missing_topic_count > 0:
                print(f"{Fore.YELLOW}发现 {missing_topic_count} 个节点缺少主题属性，正在更新...{Style.RESET_ALL}")
                
                # 获取所有缺少topic的节点
                result = session.run("""
                    MATCH (m:Memory)
                    WHERE m.topic IS NULL AND m.user_message_preview IS NOT NULL
                    RETURN m.timestamp as timestamp, m.user_message_preview as user_msg
                    LIMIT 1000
                """)
                
                updated_count = 0
                for record in result:
                    timestamp = record["timestamp"]
                    user_msg = record["user_msg"]
                    
                    # 提取主题
                    topic = extract_topic(user_msg)
                    
                    # 更新节点
                    session.run("""
                        MATCH (m:Memory {timestamp: $timestamp})
                        SET m.topic = $topic
                    """, timestamp=timestamp, topic=topic)
                    
                    updated_count += 1
                
                print(f"{Fore.GREEN}已更新 {updated_count} 个节点的主题属性{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}所有节点都已有主题属性{Style.RESET_ALL}")
                
    except Exception as e:
        print(f"{Fore.RED}更新节点主题属性时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"更新节点主题属性时发生错误: {str(e)}")

def clear_all_memories():
    """清除所有存储的记忆（Neo4j和FAISS）"""
    print(f"\n{Fore.RED}{'!' * 50}{Style.RESET_ALL}")
    print(f"{Fore.RED}{'!' * 15}{Fore.WHITE} 警告：即将清除所有记忆 {Fore.RED}{'!' * 15}{Style.RESET_ALL}")
    print(f"{Fore.RED}{'!' * 50}{Style.RESET_ALL}")
    
    confirm = input(f"{Fore.YELLOW}此操作将删除所有存储的对话记忆，无法恢复！\n确认清除? (y/n): {Style.RESET_ALL}")
    
    if confirm.lower() != 'y':
        print(f"{Fore.GREEN}操作已取消，记忆保持不变。{Style.RESET_ALL}")
        return False
    
    try:
        # 1. 清除Neo4j数据库
        print(f"\n{Fore.CYAN}1. 清除Neo4j数据库...{Style.RESET_ALL}")
        with neo4j_db.driver.session() as session:
            # 删除所有节点和关系
            result = session.run("MATCH (n) DETACH DELETE n")
            print(f"{Fore.GREEN}Neo4j数据库已清空{Style.RESET_ALL}")
        
        # 2. 清除FAISS索引
        print(f"\n{Fore.CYAN}2. 清除FAISS向量存储...{Style.RESET_ALL}")
        # 创建新的空索引
        dimension = config["storage"]["faiss"]["dimension"]
        new_index = faiss.IndexFlatL2(dimension)
        memory_store.index = new_index
        memory_store.texts = []
        memory_store.save_index()
        print(f"{Fore.GREEN}FAISS向量存储已清空{Style.RESET_ALL}")
        
        # 3. 重新初始化数据库结构
        print(f"\n{Fore.CYAN}3. 重新初始化数据库结构...{Style.RESET_ALL}")
        neo4j_db.init_database()
        
        print(f"\n{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 15}{Fore.WHITE} 所有记忆已清除 {Fore.GREEN}{'=' * 15}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        
        # 记录日志
        logger.info("\n================ 所有记忆已清除 ================\n")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}清除记忆时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"清除记忆时发生错误: {str(e)}")
        return False

def clear_memories_by_keyword(keyword: str):
    """根据关键词清除特定的记忆"""
    print(f"\n{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 10}{Fore.WHITE} 根据关键词清除记忆: '{keyword}' {Fore.YELLOW}{'=' * 10}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    
    confirm = input(f"{Fore.YELLOW}此操作将删除包含关键词 '{keyword}' 的所有记忆，无法恢复！\n确认清除? (y/n): {Style.RESET_ALL}")
    
    if confirm.lower() != 'y':
        print(f"{Fore.GREEN}操作已取消，记忆保持不变。{Style.RESET_ALL}")
        return False
    
    try:
        # 1. 从Neo4j中查找并删除包含关键词的记忆
        print(f"\n{Fore.CYAN}1. 从Neo4j中查找包含关键词的记忆...{Style.RESET_ALL}")
        with neo4j_db.driver.session() as session:
            # 查找包含关键词的节点
            result = session.run("""
                MATCH (m:Memory)
                WHERE m.user_message_preview CONTAINS $keyword OR 
                      m.ai_response_preview CONTAINS $keyword OR
                      m.topic CONTAINS $keyword
                RETURN m.timestamp as timestamp
            """, keyword=keyword)
            
            timestamps = [record["timestamp"] for record in result]
            
            if not timestamps:
                print(f"{Fore.YELLOW}未找到包含关键词 '{keyword}' 的记忆{Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}找到 {len(timestamps)} 条包含关键词的记忆{Style.RESET_ALL}")
            
            # 删除这些节点及其关系
            session.run("""
                MATCH (m:Memory)
                WHERE m.timestamp IN $timestamps
                DETACH DELETE m
            """, timestamps=timestamps)
            
            print(f"{Fore.GREEN}Neo4j中的相关记忆已删除{Style.RESET_ALL}")
        
        # 2. 从FAISS中删除相应的记忆
        print(f"\n{Fore.CYAN}2. 从FAISS中删除相应的记忆...{Style.RESET_ALL}")
        
        # 由于FAISS不支持直接删除，我们需要重建索引
        new_texts = []
        indices_to_keep = []
        
        for i, item in enumerate(memory_store.texts):
            if item["timestamp"] not in timestamps:
                new_texts.append(item)
                indices_to_keep.append(i)
        
        # 如果有记忆被删除
        if len(new_texts) < len(memory_store.texts):
            # 创建新的索引
            dimension = config["storage"]["faiss"]["dimension"]
            new_index = faiss.IndexFlatL2(dimension)
            
            # 只保留未删除的记忆
            if indices_to_keep:
                # 获取要保留的向量
                vectors_to_keep = np.array([memory_store.index.reconstruct(i) for i in indices_to_keep])
                # 添加到新索引
                new_index.add(vectors_to_keep)
            
            # 更新内存中的索引和文本
            memory_store.index = new_index
            memory_store.texts = new_texts
            memory_store.save_index()
            
            print(f"{Fore.GREEN}FAISS中的相关记忆已删除{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 10}{Fore.WHITE} 包含关键词 '{keyword}' 的记忆已清除 {Fore.GREEN}{'=' * 10}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        
        # 记录日志
        logger.info(f"\n================ 已清除包含关键词 '{keyword}' 的记忆 ================\n")
        logger.info(f"共清除 {len(timestamps)} 条记忆")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}清除记忆时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"清除记忆时发生错误: {str(e)}")
        return False

# 添加 json_serial 函数用于处理 DateTime 序列化
def json_serial(obj):
    """处理JSON序列化中的特殊类型"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# 修复备份功能中的 DateTime 序列化问题
def backup_memories(backup_path=None):
    """备份所有记忆到文件"""
    if backup_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"backups/memory_backup_{timestamp}"
    
    # 确保备份目录存在
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    
    try:
        # 1. 备份FAISS索引和文本
        print(f"\n{Fore.CYAN}1. 备份FAISS数据...{Style.RESET_ALL}")
        faiss_backup = {
            "texts": memory_store.texts,
            "index_data": faiss.serialize_index(memory_store.index)
        }
        with open(f"{backup_path}_faiss.pkl", "wb") as f:
            pickle.dump(faiss_backup, f)
        
        # 2. 备份Neo4j数据
        print(f"\n{Fore.CYAN}2. 备份Neo4j数据...{Style.RESET_ALL}")
        with neo4j_db.driver.session() as session:
            # 获取所有节点和关系
            result = session.run("""
                MATCH (n:Memory)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, collect({rel: r, target: m}) as relationships
            """)
            
            neo4j_backup = []
            for record in result:
                node = record["n"]
                relationships = record["relationships"]
                
                # 将节点属性转换为普通字典
                node_data = dict(node.items())
                rel_data = []
                
                for rel_info in relationships:
                    if rel_info["rel"] is not None:
                        # 将关系属性转换为普通字典
                        rel_properties = dict(rel_info["rel"].items())
                        target_data = dict(rel_info["target"].items()) if rel_info["target"] is not None else {}
                        
                        rel_data.append({
                            "type": rel_info["rel"].type,
                            "properties": rel_properties,
                            "target_timestamp": target_data.get("timestamp")
                        })
                
                neo4j_backup.append({
                    "node": node_data,
                    "relationships": rel_data
                })
            
            # 使用自定义序列化函数处理DateTime
            with open(f"{backup_path}_neo4j.json", "w", encoding="utf-8") as f:
                json.dump(neo4j_backup, f, default=json_serial, ensure_ascii=False, indent=2)
        
        print(f"\n{Fore.GREEN}记忆备份完成: {backup_path}_faiss.pkl 和 {backup_path}_neo4j.json{Style.RESET_ALL}")
        logger.info(f"记忆备份完成: {backup_path}")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}备份记忆时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"备份记忆时发生错误: {str(e)}")
        return False

def restore_memories(backup_path):
    """从备份文件恢复记忆"""
    if not os.path.exists(f"{backup_path}_faiss.pkl") or not os.path.exists(f"{backup_path}_neo4j.json"):
        print(f"{Fore.RED}备份文件不存在: {backup_path}{Style.RESET_ALL}")
        return False
    
    confirm = input(f"{Fore.YELLOW}此操作将覆盖当前所有记忆！\n确认恢复? (y/n): {Style.RESET_ALL}")
    if confirm.lower() != 'y':
        print(f"{Fore.GREEN}操作已取消，记忆保持不变。{Style.RESET_ALL}")
        return False
    
    try:
        # 首先清除所有现有记忆
        clear_all_memories(confirm_override=True)
        
        # 1. 恢复FAISS数据
        print(f"\n{Fore.CYAN}1. 恢复FAISS数据...{Style.RESET_ALL}")
        with open(f"{backup_path}_faiss.pkl", "rb") as f:
            faiss_backup = pickle.load(f)
        
        memory_store.texts = faiss_backup["texts"]
        memory_store.index = faiss.deserialize_index(faiss_backup["index_data"])
        memory_store.save_index()
        
        # 2. 恢复Neo4j数据
        print(f"\n{Fore.CYAN}2. 恢复Neo4j数据...{Style.RESET_ALL}")
        with open(f"{backup_path}_neo4j.json", "r", encoding="utf-8") as f:
            neo4j_backup = json.load(f)
        
        with neo4j_db.driver.session() as session:
            # 先创建所有节点
            for item in neo4j_backup:
                node = item["node"]
                properties_str = ", ".join([f"{k}: ${k}" for k in node.keys()])
                
                session.run(f"""
                    CREATE (m:Memory {{{properties_str}}})
                """, **node)
            
            # 再创建所有关系
            for item in neo4j_backup:
                source_timestamp = item["node"]["timestamp"]
                
                for rel in item["relationships"]:
                    target_timestamp = rel["target_timestamp"]
                    rel_type = rel["type"]
                    properties = rel["properties"]
                    
                    properties_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                    
                    session.run(f"""
                        MATCH (source:Memory {{timestamp: $source_timestamp}})
                        MATCH (target:Memory {{timestamp: $target_timestamp}})
                        CREATE (source)-[r:{rel_type} {{{properties_str}}}]->(target)
                    """, source_timestamp=source_timestamp, target_timestamp=target_timestamp, **properties)
        
        print(f"\n{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 15}{Fore.WHITE} 记忆恢复完成 {Fore.GREEN}{'=' * 15}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        
        logger.info(f"记忆从 {backup_path} 恢复完成")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}恢复记忆时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"恢复记忆时发生错误: {str(e)}")
        return False

def memory_statistics():
    """显示记忆统计信息"""
    try:
        # 1. FAISS统计
        faiss_count = len(memory_store.texts)
        
        # 修复：检查索引文件是否存在
        faiss_index_path = "faiss_index.pkl"
        if os.path.exists(faiss_index_path):
            faiss_size = os.path.getsize(faiss_index_path) / (1024 * 1024)  # MB
        else:
            faiss_size = 0
            print(f"{Fore.YELLOW}警告: FAISS索引文件不存在，可能是首次运行{Style.RESET_ALL}")
        
        # 2. Neo4j统计
        with neo4j_db.driver.session() as session:
            node_result = session.run("MATCH (n:Memory) RETURN count(n) as count").single()
            node_count = node_result["count"] if node_result else 0
            
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()
            rel_count = rel_result["count"] if rel_result else 0
            
            # 获取最早和最新的记忆
            earliest_result = session.run("""
                MATCH (n:Memory)
                RETURN n.timestamp as timestamp
                ORDER BY timestamp ASC
                LIMIT 1
            """).single()
            
            latest_result = session.run("""
                MATCH (n:Memory)
                RETURN n.timestamp as timestamp
                ORDER BY timestamp DESC
                LIMIT 1
            """).single()
            
            earliest = earliest_result["timestamp"] if earliest_result else "无记忆"
            latest = latest_result["timestamp"] if latest_result else "无记忆"
            
            # 获取主题分布
            topic_result = session.run("""
                MATCH (n:Memory)
                WHERE n.topic IS NOT NULL
                RETURN n.topic as topic, count(*) as count
                ORDER BY count DESC
                LIMIT 5
            """)
            
            topics = [(record["topic"], record["count"]) for record in topic_result]
        
        # 打印统计信息
        print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 15}{Fore.WHITE} 记忆统计信息 {Fore.CYAN}{'=' * 15}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}基本统计:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}FAISS记忆数量: {Fore.GREEN}{faiss_count}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}FAISS索引大小: {Fore.GREEN}{faiss_size:.2f} MB{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Neo4j节点数量: {Fore.GREEN}{node_count}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Neo4j关系数量: {Fore.GREEN}{rel_count}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}时间范围:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}最早记忆: {Fore.GREEN}{earliest}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}最新记忆: {Fore.GREEN}{latest}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}热门主题:{Style.RESET_ALL}")
        for topic, count in topics:
            print(f"{Fore.WHITE}{topic}: {Fore.GREEN}{count}条记忆{Style.RESET_ALL}")
        
        # 检查数据一致性
        print(f"\n{Fore.YELLOW}数据一致性检查:{Style.RESET_ALL}")
        if faiss_count == node_count:
            print(f"{Fore.GREEN}FAISS和Neo4j记忆数量一致 ✓{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}警告: FAISS ({faiss_count}) 和 Neo4j ({node_count}) 记忆数量不一致 ✗{Style.RESET_ALL}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}获取记忆统计信息时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"获取记忆统计信息时发生错误: {str(e)}")
        return False

def optimize_memories():
    """优化记忆存储，合并相似记忆，删除低质量记忆"""
    print(f"\n{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 15}{Fore.WHITE} 记忆优化 {Fore.YELLOW}{'=' * 15}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    
    confirm = input(f"{Fore.YELLOW}此操作将合并高度相似的记忆并删除低质量记忆。\n确认优化? (y/n): {Style.RESET_ALL}")
    if confirm.lower() != 'y':
        print(f"{Fore.GREEN}操作已取消，记忆保持不变。{Style.RESET_ALL}")
        return False
    
    try:
        # 1. 备份当前记忆
        print(f"\n{Fore.CYAN}1. 备份当前记忆...{Style.RESET_ALL}")
        backup_path = f"backups/pre_optimize_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_memories(backup_path)
        
        # 2. 查找并合并高度相似的记忆
        print(f"\n{Fore.CYAN}2. 查找高度相似的记忆...{Style.RESET_ALL}")
        with neo4j_db.driver.session() as session:
            # 查找相似度超过0.95的记忆对
            result = session.run("""
                MATCH (a:Memory)-[r:SIMILAR_TO]->(b:Memory)
                WHERE r.similarity > 0.95 AND a.timestamp < b.timestamp
                RETURN a.timestamp as older, b.timestamp as newer, r.similarity as similarity
                ORDER BY r.similarity DESC
            """)
            
            similar_pairs = [(record["older"], record["newer"], record["similarity"]) for record in result]
            
            if not similar_pairs:
                print(f"{Fore.GREEN}未找到需要合并的高度相似记忆{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}找到 {len(similar_pairs)} 对高度相似的记忆{Style.RESET_ALL}")
                
                # 记录要删除的时间戳
                to_delete = set()
                
                for older, newer, similarity in similar_pairs:
                    if newer not in to_delete:  # 只有当较新的记忆未被标记删除时才处理
                        print(f"{Fore.WHITE}合并: {older} -> {newer} (相似度: {similarity:.4f}){Style.RESET_ALL}")
                        to_delete.add(older)  # 标记较旧的记忆为删除
                
                # 删除被合并的记忆
                if to_delete:
                    timestamps = list(to_delete)
                    print(f"{Fore.WHITE}删除 {len(timestamps)} 条被合并的记忆{Style.RESET_ALL}")
                    
                    # 从Neo4j删除
                    session.run("""
                        MATCH (m:Memory)
                        WHERE m.timestamp IN $timestamps
                        DETACH DELETE m
                    """, timestamps=timestamps)
                    
                    # 从FAISS删除
                    new_texts = []
                    indices_to_keep = []
                    
                    for i, item in enumerate(memory_store.texts):
                        if item["timestamp"] not in timestamps:
                            new_texts.append(item)
                            indices_to_keep.append(i)
                    
                    # 重建FAISS索引
                    dimension = config["storage"]["faiss"]["dimension"]
                    new_index = faiss.IndexFlatL2(dimension)
                    
                    if indices_to_keep:
                        vectors_to_keep = np.array([memory_store.index.reconstruct(i) for i in indices_to_keep])
                        new_index.add(vectors_to_keep)
                    
                    memory_store.index = new_index
                    memory_store.texts = new_texts
                    memory_store.save_index()
        
        # 3. 显示优化结果
        print(f"\n{Fore.CYAN}3. 优化完成，显示结果...{Style.RESET_ALL}")
        memory_statistics()
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}优化记忆时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"优化记忆时发生错误: {str(e)}")
        return False

# 添加清除日志功能
def clear_logs():
    """清除日志文件"""
    print(f"\n{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 15}{Fore.WHITE} 清除日志文件 {Fore.YELLOW}{'=' * 15}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
    
    confirm = input(f"{Fore.YELLOW}此操作将清空所有日志文件，无法恢复！\n确认清除? (y/n): {Style.RESET_ALL}")
    if confirm.lower() != 'y':
        print(f"{Fore.GREEN}操作已取消，日志文件保持不变。{Style.RESET_ALL}")
        return False
    
    try:
        # 获取所有日志文件
        log_dir = 'logs'
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        if not log_files:
            print(f"{Fore.YELLOW}未找到日志文件{Style.RESET_ALL}")
            return False
        
        # 清除所有日志文件
        for log_file in log_files:
            file_path = os.path.join(log_dir, log_file)
            # 不删除文件，而是清空内容
            with open(file_path, 'w') as f:
                pass
            print(f"{Fore.GREEN}已清空日志文件: {log_file}{Style.RESET_ALL}")
        
        # 记录新的日志条目
        logger.info("\n================ 日志已清空 ================\n")
        
        print(f"\n{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 15}{Fore.WHITE} 所有日志已清空 {Fore.GREEN}{'=' * 15}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}清除日志时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"清除日志时发生错误: {str(e)}")
        return False

# 添加图节点内容展示功能
def visualize_memory_graph(limit=10, keyword=None):
    """Visualize the memory graph
    
    Args:
        limit: Maximum number of nodes to display
        keyword: Optional filter keyword
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        
        # Attempt to load a Chinese font
        try:
            font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
        except:
            font = FontProperties()
        
        print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 15}{Fore.WHITE} Memory Graph Visualization {Fore.CYAN}{'=' * 15}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        
        # Fetch nodes and relationships from Neo4j
        with neo4j_db.driver.session() as session:
            # Check if there are any nodes
            check_result = session.run("MATCH (n:Memory) RETURN count(n) as count").single()
            if not check_result or check_result["count"] == 0:
                print(f"{Fore.YELLOW}No memory nodes found, unable to generate graph{Style.RESET_ALL}")
                return False
            
            # Build query
            query = """
                MATCH (n:Memory)
                {}
                WITH n
                ORDER BY n.timestamp DESC
                LIMIT $limit
                OPTIONAL MATCH (n)-[r:SIMILAR_TO]->(m:Memory)
                WHERE r.similarity >= 0.7
                RETURN n, collect({rel: r, target: m}) as relationships
            """
            
            # Add filter clause if keyword is provided
            if keyword:
                filter_clause = """
                    WHERE n.user_message_preview CONTAINS $keyword OR 
                          n.ai_response_preview CONTAINS $keyword OR
                          n.topic CONTAINS $keyword
                """
                query = query.format(filter_clause)
                params = {"limit": limit, "keyword": keyword}
                print(f"{Fore.CYAN}Filtering with keyword: {Fore.YELLOW}{keyword}{Style.RESET_ALL}")
            else:
                query = query.format("")
                params = {"limit": limit}
            
            result = session.run(query, **params)
            
            # Create graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            nodes = {}
            node_count = 0
            edge_count = 0
            
            for record in result:
                node = record["n"]
                relationships = record["relationships"]
                
                # Get node attributes
                node_id = node.get("timestamp")
                if not node_id:  # Skip nodes without timestamps
                    continue
                    
                user_msg_preview = node.get("user_message_preview", "No Preview")
                
                # Add node
                if node_id not in nodes:
                    nodes[node_id] = {
                        "id": node_id,
                        "label": user_msg_preview[:20] + "..." if len(user_msg_preview) > 20 else user_msg_preview,
                        "time": node_id.split(".")[0] if "." in node_id else node_id
                    }
                    G.add_node(node_id, **nodes[node_id])
                    node_count += 1
                
                # Add edges
                for rel_info in relationships:
                    rel = rel_info.get("rel")
                    target = rel_info.get("target")
                    
                    if rel is not None and target is not None:
                        target_id = target.get("timestamp")
                        if not target_id:  # Skip target nodes without timestamps
                            continue
                            
                        similarity = rel.get("similarity", 0)
                        
                        # Ensure target node exists
                        if target_id and target_id not in nodes:
                            target_msg_preview = target.get("user_message_preview", "No Preview")
                            
                            nodes[target_id] = {
                                "id": target_id,
                                "label": target_msg_preview[:20] + "..." if len(target_msg_preview) > 20 else target_msg_preview,
                                "time": target_id.split(".")[0] if "." in target_id else target_id
                            }
                            G.add_node(target_id, **nodes[target_id])
                            node_count += 1
                        
                        # Add edge
                        if target_id:
                            G.add_edge(node_id, target_id, weight=similarity, similarity=f"{similarity:.2f}")
                            edge_count += 1
            
            if node_count == 0:
                print(f"{Fore.YELLOW}No matching memory nodes found, unable to generate graph{Style.RESET_ALL}")
                return False
                
            print(f"{Fore.GREEN}Found {node_count} nodes and {edge_count} relationships{Style.RESET_ALL}")
            
            # Set figure size
            plt.figure(figsize=(12, 10))
            
            # Use spring layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
            
            # Draw edges
            edges = G.edges(data=True)
            edge_colors = [float(e[2]['weight']) for u, v, e in edges]
            nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors, edge_cmap=plt.cm.Blues, arrows=True, arrowsize=15)
            
            # Draw node labels
            labels = {node: data['label'] for node, data in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family=font.get_name())
            
            # Draw edge labels
            edge_labels = {(u, v): f"{d['similarity']}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            # Set title
            title = f"Memory Relationship Graph - Displaying {node_count} Nodes"
            if keyword:
                title += f" (Keyword: {keyword})"
            plt.title(title, fontproperties=font, fontsize=16)
            
            # Save image
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_graph_{timestamp}.png"
            if keyword:
                filename = f"memory_graph_{keyword}_{timestamp}.png"
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"{Fore.GREEN}Graph saved as: {filename}{Style.RESET_ALL}")
            
            # Show image
            plt.show()
            
            return True
        
    except ImportError as e:
        print(f"{Fore.RED}Missing required library: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please install the required libraries: pip install networkx matplotlib{Style.RESET_ALL}")
        return False
    except Exception as e:
        print(f"{Fore.RED}Error visualizing memory graph: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Error visualizing memory graph: {str(e)}")
        return False

# 添加查看节点详情功能
def view_memory_node(timestamp=None):
    """查看特定记忆节点的详细信息
    
    Args:
        timestamp: 节点的时间戳，如果为None则提示用户输入
    """
    try:
        if timestamp is None:
            # 获取最近的记忆节点列表
            with neo4j_db.driver.session() as session:
                result = session.run("""
                    MATCH (m:Memory)
                    RETURN m.timestamp as timestamp, m.user_message_preview as preview, m.topic as topic
                    ORDER BY m.timestamp DESC
                    LIMIT 10
                """)
                
                recent_nodes = [(record["timestamp"], record["preview"], record["topic"]) for record in result]
                
                if not recent_nodes:
                    print(f"{Fore.YELLOW}没有找到记忆节点{Style.RESET_ALL}")
                    return False
                
                print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'=' * 15}{Fore.WHITE} 最近的记忆节点 {Fore.CYAN}{'=' * 15}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
                
                for i, (ts, preview, topic) in enumerate(recent_nodes, 1):
                    print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} [{ts}] {topic}: {preview[:30]}...")
                
                choice = input(f"\n{Fore.YELLOW}请选择要查看的节点编号 (1-{len(recent_nodes)}): {Style.RESET_ALL}")
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(recent_nodes):
                        timestamp = recent_nodes[index][0]
                    else:
                        print(f"{Fore.RED}无效的选择{Style.RESET_ALL}")
                        return False
                except ValueError:
                    print(f"{Fore.RED}请输入有效的数字{Style.RESET_ALL}")
                    return False
        
        # 获取节点详情
        with neo4j_db.driver.session() as session:
            # 获取节点信息
            node_result = session.run("""
                MATCH (m:Memory {timestamp: $timestamp})
                RETURN m
            """, timestamp=timestamp).single()
            
            if not node_result:
                print(f"{Fore.RED}未找到时间戳为 {timestamp} 的记忆节点{Style.RESET_ALL}")
                return False
            
            node = node_result["m"]
            
            # 获取关系信息
            rel_result = session.run("""
                MATCH (m:Memory {timestamp: $timestamp})-[r:SIMILAR_TO]->(n:Memory)
                RETURN n.timestamp as target, r.similarity as similarity
                ORDER BY r.similarity DESC
            """, timestamp=timestamp)
            
            relationships = [(record["target"], record["similarity"]) for record in rel_result]
            
            # 获取反向关系
            rev_rel_result = session.run("""
                MATCH (n:Memory)-[r:SIMILAR_TO]->(m:Memory {timestamp: $timestamp})
                RETURN n.timestamp as source, r.similarity as similarity
                ORDER BY r.similarity DESC
            """, timestamp=timestamp)
            
            rev_relationships = [(record["source"], record["similarity"]) for record in rev_rel_result]
        
        # 从FAISS获取完整内容
        memory_text = None
        for item in memory_store.texts:
            if item["timestamp"] == timestamp:
                memory_text = item["text"]
                break
        
        # 显示节点详情
        print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 15}{Fore.WHITE} 记忆节点详情 {Fore.CYAN}{'=' * 15}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}基本信息:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}时间戳: {Fore.GREEN}{timestamp}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}主题: {Fore.GREEN}{node.get('topic', '未知')}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}Neo4j存储的内容:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}用户消息预览: {Fore.GREEN}{node.get('user_message_preview', '无')}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}AI回答预览: {Fore.GREEN}{node.get('ai_response_preview', '无')}{Style.RESET_ALL}")
        
        if memory_text:
            print(f"\n{Fore.YELLOW}FAISS存储的完整内容:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{memory_text}{Style.RESET_ALL}")
        
        if relationships:
            print(f"\n{Fore.YELLOW}相似记忆 (出度关系):{Style.RESET_ALL}")
            for i, (target, similarity) in enumerate(relationships, 1):
                print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} 目标: {target} (相似度: {similarity:.4f})")
        
        if rev_relationships:
            print(f"\n{Fore.YELLOW}被引用记忆 (入度关系):{Style.RESET_ALL}")
            for i, (source, similarity) in enumerate(rev_relationships, 1):
                print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} 来源: {source} (相似度: {similarity:.4f})")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}查看记忆节点时发生错误: {str(e)}{Style.RESET_ALL}")
        logger.error(f"查看记忆节点时发生错误: {str(e)}")
        return False

def initialize_system():
    """初始化系统，检查必要的文件和目录"""
    print(f"\n{Fore.CYAN}正在初始化系统...{Style.RESET_ALL}")
    
    # 1. 检查并创建必要的目录
    directories = ['logs', 'backups', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"{Fore.GREEN}创建目录: {directory}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}创建目录 {directory} 失败: {str(e)}{Style.RESET_ALL}")
    
    # 2. 检查FAISS索引文件目录权限
    faiss_index_path = "faiss_index.pkl"
    faiss_dir = os.path.dirname(faiss_index_path) or "."
    if not os.access(faiss_dir, os.W_OK):
        print(f"{Fore.RED}警告: 没有写入权限到 {faiss_dir} 目录，FAISS索引可能无法保存{Style.RESET_ALL}")
    
    # 3. 检查必要的Python库
    try:
        import networkx
        import matplotlib
        print(f"{Fore.GREEN}可视化依赖库已安装{Style.RESET_ALL}")
    except ImportError:
        print(f"{Fore.YELLOW}警告: 缺少可视化依赖库，请安装: pip install networkx matplotlib{Style.RESET_ALL}")
    
    # 4. 检查Neo4j连接
    try:
        with neo4j_db.driver.session() as session:
            result = session.run("RETURN 1 as test").single()
            if result and result["test"] == 1:
                print(f"{Fore.GREEN}Neo4j数据库连接正常{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Neo4j数据库连接失败: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}请确保Neo4j数据库已启动，并检查连接配置{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}系统初始化完成{Style.RESET_ALL}")

try:
    print(f"\n{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'=' * 15}{Fore.WHITE} AI助手启动 {Fore.YELLOW}{'=' * 15}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'=' * 50}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}当前使用的模型: {config['model']['name']}{Style.RESET_ALL}\n")
    
    # 初始化系统
    initialize_system()
    
    # 检查并更新数据库结构
    print(f"\n{Fore.CYAN}检查并更新数据库结构...{Style.RESET_ALL}")
    update_existing_nodes_with_topic()
    
    print(f"\n{Fore.CYAN}欢迎使用持久记忆AI助手！输入'help'查看所有命令。{Style.RESET_ALL}")
    logger.info("\n================ AI助手启动 ================\n")
    total_cost = 0.0
    
    while True:
        print(f"\n{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        message = input(f"{Fore.CYAN}【用户输入】{Fore.WHITE} 请输入您的问题（特殊命令请参考下方）：\n{Fore.YELLOW}>>> {Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 50}{Style.RESET_ALL}")
        
        # 命令处理
        if message.lower() == "exit":
            break
        elif message.lower() == "clear_all":
            clear_all_memories()
            continue
        elif message.lower().startswith("clear:"):
            keyword = message[6:].strip()
            if keyword:
                clear_memories_by_keyword(keyword)
            else:
                print(f"{Fore.YELLOW}请指定要清除的关键词，例如: clear:猫娘{Style.RESET_ALL}")
            continue
        elif message.lower() == "backup":
            backup_memories()
            continue
        elif message.lower().startswith("restore:"):
            path = message[8:].strip()
            if path:
                restore_memories(path)
            else:
                print(f"{Fore.YELLOW}请指定要恢复的备份路径，例如: restore:backups/memory_backup_20250316_123456{Style.RESET_ALL}")
            continue
        elif message.lower() == "stats":
            memory_statistics()
            continue
        elif message.lower() == "optimize":
            optimize_memories()
            continue
        elif message.lower() == "help":
            print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 15}{Fore.WHITE} 命令帮助 {Fore.CYAN}{'=' * 15}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
            
            print(f"\n{Fore.YELLOW}基本命令:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}exit{Style.RESET_ALL} - 退出程序")
            print(f"{Fore.GREEN}help{Style.RESET_ALL} - 显示此帮助信息")
            
            print(f"\n{Fore.YELLOW}记忆管理:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}clear_all{Style.RESET_ALL} - 清除所有存储的记忆")
            print(f"{Fore.GREEN}clear:关键词{Style.RESET_ALL} - 清除包含特定关键词的记忆")
            print(f"{Fore.GREEN}backup{Style.RESET_ALL} - 备份当前所有记忆")
            print(f"{Fore.GREEN}restore:路径{Style.RESET_ALL} - 从指定路径恢复记忆")
            print(f"{Fore.GREEN}optimize{Style.RESET_ALL} - 优化记忆存储")
            
            print(f"\n{Fore.YELLOW}记忆可视化与查询:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}stats{Style.RESET_ALL} - 显示记忆统计信息")
            print(f"{Fore.GREEN}search{Style.RESET_ALL} 或 {Fore.GREEN}find{Style.RESET_ALL} - 搜索记忆")
            print(f"{Fore.GREEN}search:关键词{Style.RESET_ALL} - 搜索包含特定关键词的记忆")
            print(f"{Fore.GREEN}graph{Style.RESET_ALL} - 可视化记忆图谱")
            print(f"{Fore.GREEN}graph:关键词{Style.RESET_ALL} - 可视化包含特定关键词的记忆图谱")
            print(f"{Fore.GREEN}node{Style.RESET_ALL} - 查看特定记忆节点的详细信息")
            print(f"{Fore.GREEN}node:时间戳{Style.RESET_ALL} - 查看指定时间戳的记忆节点")
            
            print(f"\n{Fore.YELLOW}系统维护:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}logs:clear{Style.RESET_ALL} - 清除日志文件")
            print(f"{Fore.GREEN}init{Style.RESET_ALL} 或 {Fore.GREEN}initialize{Style.RESET_ALL} - 初始化系统检查")
            continue
        elif message.lower() == "logs:clear":
            clear_logs()
            continue
        elif message.lower() == "graph":
            visualize_memory_graph()
            continue
        elif message.lower().startswith("graph:"):
            keyword = message[6:].strip()
            if keyword:
                visualize_memory_graph(keyword=keyword)
            else:
                print(f"{Fore.YELLOW}请指定要过滤的关键词，例如: graph:猫娘{Style.RESET_ALL}")
            continue
        elif message.lower() == "node":
            view_memory_node()
            continue
        elif message.lower().startswith("node:"):
            timestamp = message[5:].strip()
            if timestamp:
                view_memory_node(timestamp)
            else:
                print(f"{Fore.YELLOW}请指定要查看的节点时间戳，例如: node:2025-03-16 12:44:15.166754{Style.RESET_ALL}")
            continue
        elif message.lower() == "search" or message.lower() == "find":
            search_memories()
            continue
        elif message.lower().startswith("search:") or message.lower().startswith("find:"):
            keyword = message[message.find(":")+1:].strip()
            if keyword:
                search_memories(keyword)
            else:
                print(f"{Fore.YELLOW}请指定要搜索的关键词，例如: search:猫娘{Style.RESET_ALL}")
            continue
        elif message.lower().startswith("export:"):
            parts = message[7:].strip().split()
            if not parts:
                export_memories()
            elif len(parts) == 1:
                if parts[0] in ["markdown", "md", "json", "txt"]:
                    export_memories(format=parts[0])
                else:
                    export_memories(keyword=parts[0])
            else:
                format_arg = parts[0] if parts[0] in ["markdown", "md", "json", "txt"] else "markdown"
                keyword_arg = parts[1] if len(parts) > 1 else None
                export_memories(format=format_arg, keyword=keyword_arg)
            continue
            
        # 记录用户输入
        logger.info("\n================ 新对话开始 ================\n")
        logger.info(f"用户输入: {message}")
        
        # 获取相关上下文
        context = get_context(message)
        
        # 构建带有上下文的提示
        print(f"\n{Fore.CYAN}构建带有上下文的提示...{Style.RESET_ALL}")
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open('prompt.md', 'r', encoding='utf-8') as file:
                prompt_content = file.read()
                print(f"\n{Fore.CYAN}读取prompt.md文件内容...{Style.RESET_ALL}")
                logger.info("读取prompt.md文件内容:")
                logger.info(prompt_content)
        except FileNotFoundError:
            logger.error("prompt.md文件不存在")
            prompt_content = "prompt.md文件不存在，无法获取内容。"
            print(f"\n{Fore.YELLOW}{prompt_content}{Style.RESET_ALL}")
        system_message = (
            basemd + "\n" +  # 在 prompt 的最前面添加 basemd 内容
            "1.你需要严格遵守的人设:"+prompt_content+"\n"
            +
            "2.你要扮演人设，根据人设回答问题，下面你与用户的对话记录，当前时间是" + current_date + "，读取然后根据对话内容和人设，再最后回复用户User问题：\n" + context
        )
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
            model=config["model"]["name"],
            messages=messages,
            stream=True,
            max_tokens=config["model"]["max_tokens"],
            temperature=config["model"]["temperature"],
            top_p=config["model"]["top_p"],
            frequency_penalty=config["model"].get("frequency_penalty", 0),
            presence_penalty=config["model"].get("presence_penalty", 0)
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
        print(f"{Fore.YELLOW}{'#' * 15}{Fore.WHITE} 对话统计 {Fore.YELLOW}{'#' * 15}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}输入tokens: {Fore.WHITE}{input_tokens:,} {Fore.YELLOW}(约￥{input_tokens * 0.000004:.4f}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}输出tokens: {Fore.WHITE}{output_tokens:,} {Fore.YELLOW}(约￥{output_tokens * 0.000016:.4f}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}本次费用: {Fore.GREEN}￥{cost:.4f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}累计总费用: {Fore.GREEN}￥{total_cost:.4f}{Style.RESET_ALL}")
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

