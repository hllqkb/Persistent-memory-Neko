from flask import Flask, render_template, request, jsonify, session, send_from_directory, Response
import os
import json
import datetime
import time
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import pickle
import faiss
from threading import Lock
from openai import OpenAI
import requests
from typing import List, Dict, Any, Tuple
import sqlite3
import glob

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = os.urandom(24)

# 配置
config = {
    "api_key": "sk-jivwbgqsesocbzkggntyzjwlkvlyhuiaphesburlvyswzsfc",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "Pro/deepseek-ai/DeepSeek-V3",
    "embedding_model": "text-embedding-3-small",
    "temperature": 0.7,
    "similarity_threshold": 0.7,
    "vector_dimension": 1024
}

# 尝试从配置文件加载
try:
    if os.path.exists('config.json'):
        with open('config.json', 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
except Exception as e:
    print(f"加载配置文件失败: {e}")

# 确保日志目录存在
if not os.path.exists('logs'):
    os.makedirs('logs')

# 设置日志
logger = logging.getLogger('neko_flask')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/neko_flask.log', maxBytes=10485760, backupCount=5, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 用于跟踪会话费用
total_cost = 0.0
cost_lock = Lock()

# 加载配置
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {
            "api_key": "sk-jivwbgqsesocbzkggntyzjwlkvlyhuiaphesburlvyswzsfc",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "Pro/deepseek-ai/DeepSeek-V3"
        }

config = load_config()

# 确保配置中有API密钥
if "api_key" not in config or not config["api_key"]:
    config["api_key"] = "sk-jivwbgqsesocbzkggntyzjwlkvlyhuiaphesburlvyswzsfc"

if "base_url" not in config or not config["base_url"]:
    config["base_url"] = "https://api.siliconflow.cn/v1"

if "model" not in config or not config["model"]:
    config["model"] = "Pro/deepseek-ai/DeepSeek-V3"

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=config["api_key"],
    base_url=config["base_url"]
)

# 内存类
class Memory:
    def __init__(self, user_message: str, ai_response: str, timestamp: str, similarity: float = None):
        self.user_message = user_message
        self.ai_response = ai_response
        self.timestamp = timestamp
        self.similarity = similarity

    def __str__(self):
        # 如果 self.timestamp 是一个浮点数
        if isinstance(self.timestamp, float):
            self.timestamp = datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
        time_str = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        similarity_str = f" [相似度: {self.similarity:.4f}]" if self.similarity is not None else ""
        return (f"[{time_str}]{similarity_str}\n"
                f"用户: {self.user_message}\n"
                f"助手: {self.ai_response[:100]}..." if len(self.ai_response) > 100 else self.ai_response)

    def to_dict(self):
        return {
            'user_message': self.user_message,
            'ai_message': self.ai_response,
            'timestamp': self.timestamp,
            'similarity': self.similarity
        }

# 向量存储类
class VectorStore:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.load_or_create_index()
        
    def load_or_create_index(self):
        """加载现有索引或创建新索引"""
        if os.path.exists('faiss_index.bin') and os.path.exists('faiss_texts.pkl'):
            try:
                self.index = faiss.read_index('faiss_index.bin')
                with open('faiss_texts.pkl', 'rb') as f:
                    self.texts = pickle.load(f)
                print(f"已加载现有索引，包含 {len(self.texts)} 条记忆")
            except Exception as e:
                logger.error(f"加载索引失败: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.texts = []
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []
        
        print(f"FAISS向量存储初始化完成，维度：{self.dimension}")
    
    def add_text(self, text: str, embedding: np.ndarray, timestamp: str):
        """添加文本及其嵌入向量到索引"""
        if self.index is None:
            self.load_or_create_index()
        
        # 添加到FAISS索引
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # 保存文本和时间戳
        self.texts.append({
            "text": text,
            "timestamp": timestamp
        })
        
        # 保存索引和文本
        faiss.write_index(self.index, 'faiss_index.bin')
        with open('faiss_texts.pkl', 'wb') as f:
            pickle.dump(self.texts, f)
    
    def search(self, embedding: np.ndarray, k: int = 5) -> List[Memory]:
        """搜索最相似的记忆"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 搜索最相似的向量
        distances, indices = self.index.search(np.array([embedding], dtype=np.float32), min(k, self.index.ntotal))
        
        # 构建结果
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

# 初始化向量存储
memory_store = VectorStore()

# SQLite数据库类
class SQLiteDatabase:
    def __init__(self, db_path='neko.db'):
        self.db_path = db_path
        self.initialize()
    
    def initialize(self):
        # 创建数据库连接
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查conversations表结构
        cursor.execute("PRAGMA table_info(conversations)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # 如果表不存在，创建表
        if not columns:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                ai_message TEXT,
                timestamp REAL,
                tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                group_id TEXT DEFAULT NULL
            )
            ''')
            logger.info("创建新的conversations表")
        # 如果表存在但列名不匹配，添加兼容性处理
        elif 'ai_response' in column_names and 'ai_message' not in column_names:
            # 使用旧的列名
            self.use_old_schema = True
            logger.info("检测到旧的数据库结构，使用兼容模式")
        else:
            self.use_old_schema = False
            logger.info("使用标准数据库结构")
            
            # 检查是否需要添加group_id列
            if 'group_id' not in column_names:
                try:
                    cursor.execute("ALTER TABLE conversations ADD COLUMN group_id TEXT DEFAULT NULL")
                    logger.info("添加group_id列到conversations表")
                except Exception as e:
                    logger.error(f"添加group_id列失败: {e}")
        
        # 检查conversation_groups表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_groups'")
        if not cursor.fetchone():
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_groups (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                updated_at REAL,
                message_count INTEGER DEFAULT 0
            )
            ''')
            logger.info("创建新的conversation_groups表")
        
        conn.commit()
        conn.close()
        
        logger.info(f"已初始化SQLite数据库: {self.db_path}")
    
    def save_conversation(self, user_message, ai_response, group_id=None, tokens=0, cost=0):
        """保存对话到指定的对话组"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = time.time()
        
        # 如果没有提供group_id，创建一个新的对话组
        if not group_id:
            # 检查是否有最近的对话组
            cursor.execute(
                "SELECT group_id FROM conversations WHERE group_id IS NOT NULL ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                group_id = row[0]
            else:
                # 创建新的对话组
                group_id = self.create_conversation_group()
        
        # 根据数据库结构选择正确的列名
        if hasattr(self, 'use_old_schema') and self.use_old_schema:
            cursor.execute(
                "INSERT INTO conversations (user_message, ai_response, timestamp, tokens, cost, group_id) VALUES (?, ?, ?, ?, ?, ?)",
                (user_message, ai_response, timestamp, tokens, cost, group_id)
            )
        else:
            cursor.execute(
                "INSERT INTO conversations (user_message, ai_message, timestamp, tokens, cost, group_id) VALUES (?, ?, ?, ?, ?, ?)",
                (user_message, ai_response, timestamp, tokens, cost, group_id)
            )
        
        # 更新对话组的消息数量和最后更新时间
        cursor.execute(
            "UPDATE conversation_groups SET message_count = message_count + 1, updated_at = ? WHERE id = ?",
            (timestamp, group_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"已保存对话，ID: {cursor.lastrowid}, 对话组: {group_id}")
        
        return timestamp, group_id
    
    def get_stats(self):
        # 获取统计信息
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总对话数
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conversation_count = cursor.fetchone()[0]
        
        # 总费用
        cursor.execute("SELECT SUM(cost) FROM conversations")
        total_cost = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'conversation_count': conversation_count,
            'total_cost': total_cost
        }
    
    def clear(self):
        # 清空数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM conversations")
        
        conn.commit()
        conn.close()
        
        logger.info("已清空数据库")

    def create_conversation_group(self, title=None):
        """创建一个新的对话组"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        group_id = str(time.time())
        created_at = time.time()
        
        # 如果没有提供标题，使用日期作为默认标题
        if not title:
            title = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute(
            "INSERT INTO conversation_groups (id, title, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?)",
            (group_id, title, created_at, created_at, 0)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"创建新的对话组: {group_id}, 标题: {title}")
        
        return group_id
    
    def get_conversation_groups(self, limit=20, offset=0):
        """获取对话组列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, title, created_at, updated_at, message_count FROM conversation_groups ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        
        groups = []
        for row in cursor.fetchall():
            group_id, title, created_at, updated_at, message_count = row
            groups.append({
                'id': group_id,
                'title': title,
                'created_at': created_at,
                'updated_at': updated_at,
                'message_count': message_count,
                'formatted_time': datetime.datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # 获取总数
        cursor.execute("SELECT COUNT(*) FROM conversation_groups")
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        return groups, total_count
    
    def get_conversation_group(self, group_id):
        """获取单个对话组信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, title, created_at, updated_at, message_count FROM conversation_groups WHERE id = ?",
            (group_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        group_id, title, created_at, updated_at, message_count = row
        group = {
            'id': group_id,
            'title': title,
            'created_at': created_at,
            'updated_at': updated_at,
            'message_count': message_count,
            'formatted_time': datetime.datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        conn.close()
        
        return group
    
    def update_conversation_group(self, group_id, title=None):
        """更新对话组信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updated_at = time.time()
        
        if title:
            cursor.execute(
                "UPDATE conversation_groups SET title = ?, updated_at = ? WHERE id = ?",
                (title, updated_at, group_id)
            )
        else:
            cursor.execute(
                "UPDATE conversation_groups SET updated_at = ? WHERE id = ?",
                (updated_at, group_id)
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"更新对话组: {group_id}")
        
        return True
    
    def delete_conversation_group(self, group_id):
        """删除对话组及其所有对话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 删除对话组中的所有对话
        cursor.execute("DELETE FROM conversations WHERE group_id = ?", (group_id,))
        
        # 删除对话组
        cursor.execute("DELETE FROM conversation_groups WHERE id = ?", (group_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"删除对话组: {group_id}")
        
        return True

# 初始化SQLite数据库
db = SQLiteDatabase()

# 获取嵌入向量
def get_embedding(text: str) -> np.ndarray:
    """使用 SiliconFlow API 获取文本嵌入向量"""
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
    
    if response.status_code != 200:
        raise Exception(f"获取嵌入向量失败: {response.text}")
    
    embedding_data = response.json()
    embedding = np.array(embedding_data["data"][0]["embedding"], dtype=np.float32)
    
    return embedding

# 计算token数和费用
def calculate_tokens_and_cost(input_text: str, output_text: str) -> Tuple[int, int, float]:
    """计算token数和费用"""
    # 简单估算token数（中文每个字约1.5个token，英文每个单词约1.3个token）
    input_tokens = len(input_text) * 1.5
    output_tokens = len(output_text) * 1.5
    
    # 计算费用（根据DeepSeek-V3的价格）
    input_cost = input_tokens * 0.000004  # 每1000个token 0.004元
    output_cost = output_tokens * 0.000016  # 每1000个token 0.016元
    total_cost = input_cost + output_cost
    
    return int(input_tokens), int(output_tokens), total_cost

# 获取上下文
def get_context(message: str) -> str:
    """获取与当前消息相关的上下文"""
    # 获取嵌入向量
    embedding = get_embedding(message)
    
    # 搜索相似记忆
    similar_memories = memory_store.search(embedding, k=5)
    
    # 构建上下文
    context = "\n\n相关记忆：\n"
    
    # 如果没有相似记忆
    if not similar_memories:
        context += "没有找到相关记忆。\n"
        return context
    
    # 添加相似对话
    context += "\n相似的历史对话：\n"
    for memory in similar_memories:
        context += f"\n[相似度: {memory.similarity:.4f}] {str(memory)}\n"
        context += "-" * 50 + "\n"
    
    return context

# 保存对话
def save_conversation(user_message: str, ai_response: str, group_id=None, tokens=0, cost=0) -> Tuple[str, str]:
    """保存对话并建立关系"""
    # 获取相似记忆
    combined_text = f"用户: {user_message}\n助手: {ai_response}"
    embedding = get_embedding(combined_text)
    similar_memories = memory_store.search(embedding, k=5)
    
    # 检查是否存在高度相似的记忆
    if similar_memories and similar_memories[0].similarity > 0.95:
        logger.info(f"检测到重复问题，相似度: {similar_memories[0].similarity:.4f}")
        logger.info(f"原问题: {similar_memories[0].user_message}")
        logger.info(f"新问题: {user_message}")
        return similar_memories[0].timestamp, group_id
    
    # 保存到SQLite
    timestamp, saved_group_id = db.save_conversation(user_message, ai_response, group_id, tokens, cost)
    
    # 保存到FAISS
    memory_store.add_text(combined_text, embedding, timestamp)
    
    return timestamp, saved_group_id

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 控制台页面路由
@app.route('/console')
def console():
    return render_template('console.html')

# 聊天API
@app.route('/api/chat', methods=['POST'])
def chat():
    global total_cost
    
    data = request.json
    message = data.get('message', '')
    conversation_timestamp = data.get('conversation_timestamp')
    group_id = data.get('group_id')
    
    if not message.strip():
        return jsonify({'error': '消息不能为空'}), 400
    
    def generate():
        global total_cost
        # 声明group_id为非局部变量，这样可以在内部函数中修改外部函数的变量
        nonlocal group_id
        try:
            # 获取相关上下文
            context = ""
            
            # 如果提供了对话组ID，获取该对话组的上下文
            if group_id:
                # 获取配置中的最大对话轮数
                max_turns = config.get("maxConversationTurns", 10)
                
                # 连接数据库
                conn = sqlite3.connect('neko.db')
                cursor = conn.cursor()
                
                # 检查表结构
                cursor.execute("PRAGMA table_info(conversations)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # 确定使用哪个字段名
                ai_field = 'ai_response' if 'ai_response' in column_names else 'ai_message'
                
                # 获取指定对话组的对话（最多max_turns轮）
                query = f"""
                    SELECT user_message, {ai_field}
                    FROM conversations 
                    WHERE group_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                cursor.execute(query, (group_id, max_turns))
                previous_conversations = cursor.fetchall()
                
                # 构建上下文（按时间正序）
                if previous_conversations:
                    context = "以下是之前的对话内容，请在回答时考虑这些上下文：\n\n"
                    for user_msg, ai_msg in reversed(previous_conversations):
                        context += f"用户: {user_msg}\n"
                        context += f"AI: {ai_msg}\n\n"
                
                conn.close()
            # 如果提供了对话时间戳，获取该对话的上下文
            elif conversation_timestamp:
                # 获取配置中的最大对话轮数
                max_turns = config.get("maxConversationTurns", 10)
                
                # 连接数据库
                conn = sqlite3.connect('neko.db')
                cursor = conn.cursor()
                
                # 检查表结构
                cursor.execute("PRAGMA table_info(conversations)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # 确定使用哪个字段名
                ai_field = 'ai_response' if 'ai_response' in column_names else 'ai_message'
                
                # 获取时间戳对应的对话组ID
                cursor.execute(
                    "SELECT group_id FROM conversations WHERE timestamp = ?",
                    (conversation_timestamp,)
                )
                row = cursor.fetchone()
                current_group_id = row[0] if row else None
                
                if current_group_id:
                    # 获取指定对话组的对话（最多max_turns轮）
                    query = f"""
                        SELECT user_message, {ai_field}
                        FROM conversations 
                        WHERE group_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """
                    cursor.execute(query, (current_group_id, max_turns))
                    previous_conversations = cursor.fetchall()
                    
                    # 构建上下文（按时间正序）
                    if previous_conversations:
                        context = "以下是之前的对话内容，请在回答时考虑这些上下文：\n\n"
                        for user_msg, ai_msg in reversed(previous_conversations):
                            context += f"用户: {user_msg}\n"
                            context += f"AI: {ai_msg}\n\n"
                    
                    # 保存当前对话组ID
                    group_id = current_group_id
                else:
                    # 如果没有找到对话组ID，使用原有的上下文获取方法
                    context = get_context(message)
                
                conn.close()
            else:
                # 如果没有提供时间戳和对话组ID，使用原有的上下文获取方法
                context = get_context(message)
            
            # 构建带有上下文的提示
            system_message = "你是一个有记忆能力的AI助手。" + context
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ]
            
            # 记录完整prompt
            logger.info("完整Prompt:")
            logger.info(f"System: {system_message}")
            logger.info(f"User: {message}")

            # 获取AI响应（使用stream=True）
            start_time = time.time()
            full_response = ""
            
            # 使用流式响应
            for response in client.chat.completions.create(
                model=config["model"],
                messages=messages,
                max_tokens=4096,
                stream=True  # 启用流式输出
            ):
                if response.choices[0].delta.content is not None:
                    content = response.choices[0].delta.content
                    full_response += content
                    # 发送数据块
                    yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"

            # 计算token数和费用
            input_tokens, output_tokens, cost = calculate_tokens_and_cost(
                system_message + message, 
                full_response
            )
            
            with cost_lock:
                total_cost += cost
            
            # 保存对话
            timestamp, saved_group_id = save_conversation(message, full_response, group_id, input_tokens + output_tokens, cost)
            
            # 发送完成标记和统计信息
            stats_data = {
                'content': '', 
                'done': True, 
                'stats': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': cost,
                    'total_cost': total_cost,
                    'time': time.time() - start_time,
                    'timestamp': timestamp,
                    'group_id': saved_group_id
                }
            }
            yield f"data: {json.dumps(stats_data)}\n\n"
            
        except Exception as e:
            logger.error(f"处理聊天请求时出错: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# 获取相似记忆API
@app.route('/api/memories', methods=['POST'])
def get_memories():
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    
    if not query.strip():
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        # 获取嵌入向量
        embedding = get_embedding(query)
        
        # 搜索相似记忆
        similar_memories = memory_store.search(embedding, k=k)
        
        # 转换为可序列化格式
        memories = []
        for memory in similar_memories:
            memories.append({
                'user_message': memory.user_message,
                'ai_response': memory.ai_response,
                'timestamp': memory.timestamp,
                'similarity': memory.similarity
            })
        
        return jsonify({'memories': memories})
        
    except Exception as e:
        logger.error(f"获取相似记忆时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 清除记忆API
@app.route('/api/clear_memory', methods=['POST'])
def clear_memory():
    try:
        # 清除向量存储
        memory_store.clear_memory()
        
        # 清除SQLite数据库
        db.clear()
        
        return jsonify({'success': True, 'message': '所有记忆已清除'})
        
    except Exception as e:
        logger.error(f"清除记忆时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 获取统计信息API
@app.route('/api/stats', methods=['GET'])
def get_stats():
    global total_cost
    try:
        # 获取SQLite中的记忆数量
        stats = db.get_stats()
        
        # 获取总费用
        with cost_lock:
            current_total_cost = total_cost
        
        return jsonify({
            'conversation_count': stats['conversation_count'],
            'total_cost': current_total_cost
        })
        
    except Exception as e:
        logger.error(f"获取统计信息时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 获取日志API
@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        logs = []
        log_files = glob.glob('logs/*.log')
        
        # 如果没有日志文件，返回空列表
        if not log_files:
            return jsonify({'logs': []})
        
        # 获取最新的日志文件
        latest_log = max(log_files, key=os.path.getmtime)
        
        # 读取日志文件
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析日志行
        for line in lines[-100:]:  # 只返回最后100行
            try:
                # 解析日志格式：时间 - 名称 - 级别 - 消息
                parts = line.strip().split(' - ', 3)
                if len(parts) >= 4:
                    timestamp_str, name, level, message = parts
                    # 转换时间戳
                    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    logs.append({
                        'timestamp': timestamp.timestamp(),
                        'name': name,
                        'level': level.lower(),
                        'message': message
                    })
            except Exception as e:
                logger.error(f"解析日志行失败: {e}")
        
        return jsonify({'logs': logs})
    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        return jsonify({'logs': [], 'error': str(e)})

# 获取对话历史API
@app.route('/api/chat_history', methods=['GET'])
def get_chat_history():
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # 连接数据库
        conn = sqlite3.connect('neko.db')
        cursor = conn.cursor()
        
        # 检查表结构
        cursor.execute("PRAGMA table_info(conversations)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # 确定使用哪个字段名
        ai_field = 'ai_response' if 'ai_response' in column_names else 'ai_message'
        
        # 获取总记录数
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_count = cursor.fetchone()[0]
        
        # 计算总页数
        total_pages = (total_count + per_page - 1) // per_page
        
        # 获取指定页的对话记录
        offset = (page - 1) * per_page
        query = f"""
            SELECT user_message, {ai_field}, timestamp, tokens, cost 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, (per_page, offset))
        
        # 构建结果
        conversations = []
        for row in cursor.fetchall():
            user_message, ai_message, timestamp, tokens, cost = row
            conversations.append({
                'user_message': user_message,
                'ai_message': ai_message,
                'timestamp': timestamp,
                'tokens': tokens,
                'cost': cost,
                'formatted_time': datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        conn.close()
        
        return jsonify({
            'conversations': conversations,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_count': total_count
            }
        })
        
    except Exception as e:
        logger.error(f"获取对话历史时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 获取完整对话历史API
@app.route('/api/full_conversation', methods=['GET'])
def get_full_conversation():
    try:
        # 获取时间戳参数
        timestamp = request.args.get('timestamp', type=float)
        if not timestamp:
            return jsonify({'error': '缺少时间戳参数'}), 400
            
        # 获取前后对话数量
        before = request.args.get('before', 0, type=int)
        after = request.args.get('after', 0, type=int)
        
        # 如果请求上下文但未指定数量，则使用配置文件中的设置
        if before == -1 or after == -1:
            max_turns = config.get("maxConversationTurns", 10)
            before = max_turns // 2
            after = max_turns // 2
        
        # 连接数据库
        conn = sqlite3.connect('neko.db')
        cursor = conn.cursor()
        
        # 检查表结构
        cursor.execute("PRAGMA table_info(conversations)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # 确定使用哪个字段名
        ai_field = 'ai_response' if 'ai_response' in column_names else 'ai_message'
        
        # 构建结果
        conversations = []
        
        # 如果before和after都为0，则只获取精确匹配时间戳的对话
        if before == 0 and after == 0:
            query = f"""
                SELECT user_message, {ai_field}, timestamp, tokens, cost 
                FROM conversations 
                WHERE timestamp = ?
            """
            cursor.execute(query, (timestamp,))
            row = cursor.fetchone()
            
            if row:
                user_message, ai_message, ts, tokens, cost = row
                conversations.append({
                    'user_message': user_message,
                    'ai_message': ai_message,
                    'timestamp': ts,
                    'tokens': tokens,
                    'cost': cost,
                    'formatted_time': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                })
        else:
            # 获取指定时间戳前后的对话
            query = f"""
                SELECT user_message, {ai_field}, timestamp, tokens, cost 
                FROM conversations 
                WHERE timestamp <= ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            cursor.execute(query, (timestamp, before))
            before_conversations = cursor.fetchall()
            
            query = f"""
                SELECT user_message, {ai_field}, timestamp, tokens, cost 
                FROM conversations 
                WHERE timestamp > ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            """
            cursor.execute(query, (timestamp, after))
            after_conversations = cursor.fetchall()
            
            # 添加之前的对话（按时间正序）
            for row in reversed(before_conversations):
                user_message, ai_message, ts, tokens, cost = row
                conversations.append({
                    'user_message': user_message,
                    'ai_message': ai_message,
                    'timestamp': ts,
                    'tokens': tokens,
                    'cost': cost,
                    'formatted_time': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # 添加之后的对话
            for row in after_conversations:
                user_message, ai_message, ts, tokens, cost = row
                conversations.append({
                    'user_message': user_message,
                    'ai_message': ai_message,
                    'timestamp': ts,
                    'tokens': tokens,
                    'cost': cost,
                    'formatted_time': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        conn.close()
        
        return jsonify({
            'conversations': conversations,
            'center_timestamp': timestamp
        })
        
    except Exception as e:
        logger.error(f"获取完整对话历史时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 获取对话组列表API
@app.route('/api/conversation_groups', methods=['GET'])
def get_conversation_groups():
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # 计算偏移量
        offset = (page - 1) * per_page
        
        # 获取对话组列表
        groups, total_count = db.get_conversation_groups(per_page, offset)
        
        # 计算总页数
        total_pages = (total_count + per_page - 1) // per_page
        
        return jsonify({
            'groups': groups,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_count': total_count
            }
        })
        
    except Exception as e:
        logger.error(f"获取对话组列表时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 获取对话组内容API
@app.route('/api/conversation_group/<group_id>', methods=['GET'])
def get_conversation_group_content(group_id):
    try:
        # 获取对话组信息
        group = db.get_conversation_group(group_id)
        if not group:
            return jsonify({'error': '对话组不存在'}), 404
        
        # 连接数据库
        conn = sqlite3.connect('neko.db')
        cursor = conn.cursor()
        
        # 检查表结构
        cursor.execute("PRAGMA table_info(conversations)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # 确定使用哪个字段名
        ai_field = 'ai_response' if 'ai_response' in column_names else 'ai_message'
        
        # 获取对话组内的所有对话
        query = f"""
            SELECT user_message, {ai_field}, timestamp, tokens, cost 
            FROM conversations 
            WHERE group_id = ? 
            ORDER BY timestamp ASC
        """
        cursor.execute(query, (group_id,))
        
        # 构建结果
        conversations = []
        for row in cursor.fetchall():
            user_message, ai_message, timestamp, tokens, cost = row
            conversations.append({
                'user_message': user_message,
                'ai_message': ai_message,
                'timestamp': timestamp,
                'tokens': tokens,
                'cost': cost,
                'formatted_time': datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        conn.close()
        
        return jsonify({
            'group': group,
            'conversations': conversations
        })
        
    except Exception as e:
        logger.error(f"获取对话组内容时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 创建对话组API
@app.route('/api/conversation_groups', methods=['POST'])
def create_conversation_group():
    try:
        data = request.json
        title = data.get('title')
        
        # 创建新的对话组
        group_id = db.create_conversation_group(title)
        
        # 获取对话组信息
        group = db.get_conversation_group(group_id)
        
        return jsonify({
            'success': True,
            'group': group
        })
        
    except Exception as e:
        logger.error(f"创建对话组时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 更新对话组API
@app.route('/api/conversation_groups/<group_id>', methods=['PUT'])
def update_conversation_group(group_id):
    try:
        data = request.json
        title = data.get('title')
        
        # 更新对话组
        db.update_conversation_group(group_id, title)
        
        # 获取更新后的对话组信息
        group = db.get_conversation_group(group_id)
        
        return jsonify({
            'success': True,
            'group': group
        })
        
    except Exception as e:
        logger.error(f"更新对话组时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 删除对话组API
@app.route('/api/conversation_groups/<group_id>', methods=['DELETE'])
def delete_conversation_group(group_id):
    try:
        # 删除对话组
        db.delete_conversation_group(group_id)
        
        return jsonify({
            'success': True,
            'message': '对话组已删除'
        })
        
    except Exception as e:
        logger.error(f"删除对话组时出错: {e}")
        return jsonify({'error': str(e)}), 500

# 路由：静态文件
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    try:
        print("启动Flask Web服务...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        print("关闭服务...") 