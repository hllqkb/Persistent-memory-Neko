import os
import yaml
import json
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """应用配置类"""
    # API配置
    API_KEY: str = Field("sk-jivwbgqsesocbzkggntyzjwlkvlyhuiaphesburlvyswzsfc", env="API_KEY")
    API_BASE_URL: str = Field("https://api.siliconflow.cn/v1", env="API_BASE_URL")
    API_TIMEOUT: int = Field(30, env="API_TIMEOUT")
    
    # 模型配置
    MODEL_NAME: str = Field("Pro/deepseek-ai/DeepSeek-V3", env="MODEL_NAME")
    MODEL_TEMPERATURE: float = Field(0.7, env="MODEL_TEMPERATURE")
    MODEL_MAX_TOKENS: int = Field(4096, env="MODEL_MAX_TOKENS")
    MODEL_TOP_P: float = Field(0.9, env="MODEL_TOP_P")
    MODEL_FREQUENCY_PENALTY: float = Field(0, env="MODEL_FREQUENCY_PENALTY")
    MODEL_PRESENCE_PENALTY: float = Field(0, env="MODEL_PRESENCE_PENALTY")
    
    # 嵌入模型配置
    EMBEDDING_MODEL: str = Field("BAAI/bge-large-zh-v1.5", env="EMBEDDING_MODEL")
    EMBEDDING_TIMEOUT: int = Field(30, env="EMBEDDING_TIMEOUT")
    
    # 重排序配置
    RERANK_ENABLED: bool = Field(True, env="RERANK_ENABLED")
    RERANK_MODEL: str = Field("BAAI/bge-reranker-v2-m3", env="RERANK_MODEL")
    RERANK_TOP_N: int = Field(5, env="RERANK_TOP_N")
    
    # 检索配置
    RETRIEVAL_GRAPH_RELATED_DEPTH: int = Field(2, env="RETRIEVAL_GRAPH_RELATED_DEPTH")
    RETRIEVAL_MIN_SIMILARITY: float = Field(0.7, env="RETRIEVAL_MIN_SIMILARITY")
    RETRIEVAL_FILTER_SIMILARITY_THRESHOLD: float = Field(0.8, env="RETRIEVAL_FILTER_SIMILARITY_THRESHOLD")
    
    # 存储配置
    NEO4J_URI: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field("neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field("12345678", env="NEO4J_PASSWORD")
    FAISS_DIMENSION: int = Field(1024, env="FAISS_DIMENSION")
    FAISS_INDEX_TYPE: str = Field("flat", env="FAISS_INDEX_TYPE")
    
    # 应用配置
    APP_NAME: str = Field("Neko API", env="APP_NAME")
    APP_VERSION: str = Field("1.0.0", env="APP_VERSION")
    APP_DESCRIPTION: str = Field("持久记忆AI助手API", env="APP_DESCRIPTION")
    DEBUG: bool = Field(False, env="DEBUG")
    
    # 文件路径
    BASE_MD_PATH: str = Field("base.md", env="BASE_MD_PATH")
    PROMPT_MD_PATH: str = Field("prompt.md", env="PROMPT_MD_PATH")
    LOGS_DIR: str = Field("logs", env="LOGS_DIR")
    BACKUPS_DIR: str = Field("backups", env="BACKUPS_DIR")
    FAISS_INDEX_PATH: str = Field("data/faiss_index.pkl", env="FAISS_INDEX_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def load_from_file(self) -> "Settings":
        """从配置文件加载配置"""
        try:
            # 首先尝试加载 YAML 格式
            try:
                with open('config.yaml', 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config:
                        print("YAML配置文件加载成功")
                        self._update_from_dict(config)
                        return self
            except Exception as yaml_error:
                print(f"加载YAML配置失败: {str(yaml_error)}, 尝试加载JSON配置")
            
            # 尝试加载标准 JSON
            try:
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print("JSON配置文件加载成功")
                    self._update_from_dict(config)
                    return self
            except Exception as json_error:
                print(f"加载JSON配置失败: {str(json_error)}")
                
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
        
        return self
    
    def _update_from_dict(self, config: Dict[str, Any]) -> None:
        """从字典更新配置"""
        if "api" in config:
            api_config = config["api"]
            if "key" in api_config:
                self.API_KEY = api_config["key"]
            if "base_url" in api_config:
                self.API_BASE_URL = api_config["base_url"]
            if "timeout" in api_config:
                self.API_TIMEOUT = api_config["timeout"]
        
        if "model" in config:
            model_config = config["model"]
            if "name" in model_config:
                self.MODEL_NAME = model_config["name"]
            if "temperature" in model_config:
                self.MODEL_TEMPERATURE = model_config["temperature"]
            if "max_tokens" in model_config:
                self.MODEL_MAX_TOKENS = model_config["max_tokens"]
            if "top_p" in model_config:
                self.MODEL_TOP_P = model_config["top_p"]
            if "frequency_penalty" in model_config:
                self.MODEL_FREQUENCY_PENALTY = model_config["frequency_penalty"]
            if "presence_penalty" in model_config:
                self.MODEL_PRESENCE_PENALTY = model_config["presence_penalty"]
        
        if "embedding" in config:
            embedding_config = config["embedding"]
            if "model" in embedding_config:
                self.EMBEDDING_MODEL = embedding_config["model"]
            if "timeout" in embedding_config:
                self.EMBEDDING_TIMEOUT = embedding_config["timeout"]
        
        if "rerank" in config:
            rerank_config = config["rerank"]
            if "enabled" in rerank_config:
                self.RERANK_ENABLED = rerank_config["enabled"]
            if "model" in rerank_config:
                self.RERANK_MODEL = rerank_config["model"]
            if "top_n" in rerank_config:
                self.RERANK_TOP_N = rerank_config["top_n"]
        
        if "retrieval" in config:
            retrieval_config = config["retrieval"]
            if "graph_related_depth" in retrieval_config:
                self.RETRIEVAL_GRAPH_RELATED_DEPTH = retrieval_config["graph_related_depth"]
            if "min_similarity" in retrieval_config:
                self.RETRIEVAL_MIN_SIMILARITY = retrieval_config["min_similarity"]
            if "filter_similarity_threshold" in retrieval_config:
                self.RETRIEVAL_FILTER_SIMILARITY_THRESHOLD = retrieval_config["filter_similarity_threshold"]
        
        if "storage" in config:
            storage_config = config["storage"]
            if "neo4j" in storage_config:
                neo4j_config = storage_config["neo4j"]
                if "uri" in neo4j_config:
                    self.NEO4J_URI = neo4j_config["uri"]
                if "user" in neo4j_config:
                    self.NEO4J_USER = neo4j_config["user"]
                if "password" in neo4j_config:
                    self.NEO4J_PASSWORD = neo4j_config["password"]
            
            if "faiss" in storage_config:
                faiss_config = storage_config["faiss"]
                if "dimension" in faiss_config:
                    self.FAISS_DIMENSION = faiss_config["dimension"]
                if "index_type" in faiss_config:
                    self.FAISS_INDEX_TYPE = faiss_config["index_type"]

# 创建全局设置实例
settings = Settings().load_from_file()

# 确保必要的目录存在
os.makedirs(settings.LOGS_DIR, exist_ok=True)
os.makedirs(settings.BACKUPS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True) 