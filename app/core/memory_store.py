import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from app.core.config import settings
from app.utils.logger import logger
from app.models.memory import Memory

class FAISSMemoryStore:
    def __init__(self, dimension=None, index_type=None, index_path=None):
        self.dimension = dimension or settings.FAISS_DIMENSION
        self.index_type = index_type or settings.FAISS_INDEX_TYPE
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.texts = []
        
        # 尝试加载现有索引
        if os.path.exists(self.index_path):
            try:
                logger.info(f"加载FAISS索引文件: {self.index_path}")
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.index = data['index']
                    self.texts = data.get('texts', [])
                logger.info(f"FAISS索引加载成功，包含 {len(self.texts)} 条记忆")
            except Exception as e:
                logger.error(f"加载FAISS索引失败: {str(e)}")
                self._create_new_index()
        else:
            logger.info(f"FAISS索引文件不存在，创建新索引")
            self._create_new_index()
            # 立即保存空索引，避免下次启动时再次提示
            self.save_index()
            logger.info(f"已创建并保存新的FAISS索引")
    
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
            logger.info(f"FAISS索引已保存，包含 {len(self.texts)} 条记忆")
            return True
        except Exception as e:
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
        logger.info(f"已保存新记忆，当前共有 {len(self.texts)} 条记忆")

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
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        logger.info("已清除所有FAISS记忆数据")

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

    def get_statistics(self) -> Dict[str, Any]:
        """获取FAISS存储统计信息"""
        stats = {
            "count": len(self.texts),
            "size_mb": 0
        }
        
        if os.path.exists(self.index_path):
            stats["size_mb"] = os.path.getsize(self.index_path) / (1024 * 1024)
        
        return stats

# 创建全局FAISS存储实例
memory_store = FAISSMemoryStore() 