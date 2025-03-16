from flask import Flask, render_template, request, jsonify, session, send_from_directory
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
        
        # 检查表结构
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
                cost REAL DEFAULT 0
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
        
        conn.commit()
        conn.close()
        
        logger.info(f"已初始化SQLite数据库: {self.db_path}")
    
    def save_conversation(self, user_message, ai_message, tokens=0, cost=0):
        # 保存对话
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = time.time()
        
        # 根据数据库结构选择正确的列名
        if hasattr(self, 'use_old_schema') and self.use_old_schema:
            cursor.execute(
                "INSERT INTO conversations (user_message, ai_response, timestamp, tokens, cost) VALUES (?, ?, ?, ?, ?)",
                (user_message, ai_message, timestamp, tokens, cost)
            )
        else:
            cursor.execute(
                "INSERT INTO conversations (user_message, ai_message, timestamp, tokens, cost) VALUES (?, ?, ?, ?, ?)",
                (user_message, ai_message, timestamp, tokens, cost)
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"已保存对话，ID: {cursor.lastrowid}")
        
        return timestamp
    
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
def save_conversation(user_message: str, ai_response: str) -> str:
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
        return similar_memories[0].timestamp
    
    # 保存到SQLite
    timestamp = db.save_conversation(user_message, ai_response)
    
    # 保存到FAISS
    memory_store.add_text(combined_text, embedding, timestamp)
    
    return timestamp

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
    
    if not message.strip():
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    try:
        # 获取相关上下文
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

        # 获取AI响应
        start_time = time.time()
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            max_tokens=4096
        )
        
        full_response = response.choices[0].message.content
        
        # 计算token数和费用
        input_tokens, output_tokens, cost = calculate_tokens_and_cost(
            system_message + message, 
            full_response
        )
        
        with cost_lock:
            total_cost += cost
        
        # 保存对话
        save_conversation(message, full_response)
        
        # 返回响应
        return jsonify({
            'response': full_response,
            'stats': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': cost,
                'total_cost': total_cost,
                'time': time.time() - start_time
            }
        })
        
    except Exception as e:
        logger.error(f"处理聊天请求时出错: {e}")
        return jsonify({'error': str(e)}), 500

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