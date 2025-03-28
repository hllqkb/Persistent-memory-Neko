import numpy as np
import requests
from typing import List, Dict, Any, Union
from app.core.config import settings
from app.utils.logger import logger

def get_embedding(text: str) -> np.ndarray:
    """使用 API 获取文本嵌入向量"""
    if not text or not isinstance(text, str):
        raise ValueError("输入文本不能为空且必须是字符串类型")
        
    # 清理和预处理文本
    text = text.strip()
    if not text:
        raise ValueError("输入文本不能全为空白字符")
    
    # 检查文本长度，如果过长则截断
    # 中文每个字约1.5个token，8192 tokens约等于5000个字符
    max_chars = 5000
    if len(text) > max_chars:
        logger.warning(f"文本过长 ({len(text)} 字符)，截断至 {max_chars} 字符")
        text = text[:max_chars]
    
    # 准备API请求
    headers = {
        "Authorization": f"Bearer {settings.API_KEY}",
        "Content-Type": "application/json"
    }
    
    # API请求数据
    data = {
        "model": settings.EMBEDDING_MODEL,
        "input": text,
        "encoding_format": "float"
    }
    
    try:
        # 发送请求
        response = requests.post(
            f"{settings.API_BASE_URL}/embeddings",
            headers=headers,
            json=data,
            timeout=settings.EMBEDDING_TIMEOUT
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
        if 'response' in locals() and response and hasattr(response, 'text'):
            logger.error(f"API响应: {response.text}")
        raise Exception(f"获取embedding失败: {str(e)}")

def rerank_documents(query: str, documents: List[str], top_n: int = None) -> List[Dict[str, Any]]:
    """使用重排序API对文档进行重排序
    
    Args:
        query: 查询文本
        documents: 候选文档列表
        top_n: 返回的最大文档数量，默认返回所有文档
        
    Returns:
        List[Dict[str, Any]]: 重排序后的文档列表，包含索引和相关性分数
    """
    if not documents:
        return []
    
    # 如果重排序功能被禁用，返回空列表
    if not settings.RERANK_ENABLED:
        return []
    
    # 准备API请求
    headers = {
        "Authorization": f"Bearer {settings.API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 设置top_n，如果未指定则使用文档数量
    if top_n is None:
        top_n = len(documents)
    
    # API请求数据
    data = {
        "model": settings.RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,  # 不需要返回文档内容
        "max_chunks_per_doc": 1024,
        "overlap_tokens": 80
    }
    
    try:
        # 发送请求
        response = requests.post(
            f"{settings.API_BASE_URL}/rerank",
            headers=headers,
            json=data,
            timeout=settings.API_TIMEOUT
        )
        
        # 检查响应状态
        if response.status_code != 200:
            error_msg = f"重排序API请求失败 (状态码: {response.status_code}): {response.text}"
            logger.error(error_msg)
            return []
        
        # 解析响应
        result = response.json()
        
        # 返回重排序结果
        return result.get("results", [])
        
    except Exception as e:
        logger.error(f"重排序过程中发生错误: {str(e)}")
        return [] 