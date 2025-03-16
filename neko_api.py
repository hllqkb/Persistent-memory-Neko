"""
neko_api.py - 提供对neko.py功能的API访问，而不运行其主程序
"""

import sys
import os
import importlib.util
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import faiss
import pickle
import json
import datetime
import time
import requests
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# 设置日志
def setup_logger():
    """配置日志记录器"""
    logger = logging.getLogger('neko_api')
    logger.setLevel(logging.INFO)
    
    # 创建 logs 目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 设置日志文件，使用 RotatingFileHandler 进行日志轮转
    handler = RotatingFileHandler(
        'logs/neko_api.log',
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

# 从neko.py导入类和函数
from neko import (
    Memory,
    Neo4jDatabase,
    VectorStore,
    get_embedding,
    calculate_tokens_and_cost
)

# 初始化 OpenAI 客户端
def get_openai_client(api_key=None, base_url=None):
    # 加载配置
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            api_key = api_key or config.get("api_key", "")
            base_url = base_url or config.get("base_url", "https://api.siliconflow.cn/v1")
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
    
    # 如果没有提供API密钥，尝试从neko.py获取
    if not api_key:
        from neko import client as neko_client
        api_key = neko_client.api_key
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

# 初始化数据库和向量存储
neo4j_db = Neo4jDatabase()
memory_store = VectorStore()

# 获取上下文函数
def get_context(message: str) -> str:
    """获取与当前消息相关的上下文"""
    logger.info(f"获取上下文: {message}")
    
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

# 保存对话函数
def save_conversation(user_message: str, ai_response: str):
    """保存对话并记录日志"""
    logger.info("保存对话...")
    
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
            logger.info("跳过保存")
            return
    
    # 保存到Neo4j并建立关系
    logger.info("1. 保存到Neo4j并建立记忆关系...")
    timestamp = neo4j_db.create_memory_with_relations(user_message, ai_response, similar_memories)
    
    # 保存到FAISS
    logger.info("2. 保存到FAISS向量存储...")
    memory_store.add_text(combined_text, embedding, timestamp)
    
    # 计算费用
    input_tokens, output_tokens, cost = calculate_tokens_and_cost(user_message, ai_response)
    logger.info(f"本次费用: ￥{cost:.7f}")
    
    # 记录日志
    logger.info("\n============ 新对话已保存 ============\n")
    logger.info(f"时间戳: {timestamp}")
    
    return timestamp

# 清理函数
def cleanup():
    """清理资源"""
    neo4j_db.close()
    logger.info("Neo4j连接已关闭") 