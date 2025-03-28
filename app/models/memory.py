from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class Memory(BaseModel):
    """记忆模型"""
    user_message: str
    ai_response: str
    timestamp: str
    similarity: Optional[float] = None
    topic: Optional[str] = None
    
    def __str__(self) -> str:
        time_str = datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        similarity_str = f" [相似度: {self.similarity:.4f}]" if self.similarity is not None else ""
        return (f"[{time_str}]{similarity_str}\n"
                f"用户: {self.user_message}\n"
                f"助手: {self.ai_response[:100]}..." if len(self.ai_response) > 100 else self.ai_response)

    def short_str(self) -> str:
        """返回简短的记忆描述"""
        time_str = datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
        similarity_str = f" [相似度: {self.similarity:.4f}]" if self.similarity is not None else ""
        return f"[{time_str}]{similarity_str}\n  问: {self.user_message[:50]}...\n  答: {self.ai_response[:50]}..."

class MemoryCreate(BaseModel):
    """创建记忆的请求模型"""
    user_message: str
    ai_response: str

class MemoryResponse(BaseModel):
    """记忆响应模型"""
    timestamp: str
    user_message: str
    ai_response: str
    topic: Optional[str] = None
    similarity: Optional[float] = None

class MemorySearchRequest(BaseModel):
    """记忆搜索请求"""
    keyword: str
    limit: int = 10

class MemorySearchResponse(BaseModel):
    """记忆搜索响应"""
    results: List[MemoryResponse]
    count: int

class MemoryStatistics(BaseModel):
    """记忆统计信息"""
    faiss_count: int
    faiss_size: float  # MB
    neo4j_node_count: int
    neo4j_rel_count: int
    earliest_memory: str
    latest_memory: str
    top_topics: List[Dict[str, Any]]
    is_consistent: bool 