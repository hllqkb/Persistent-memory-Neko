# Neko AI - 持久记忆助手

Neko AI是一个具有持久记忆能力的AI助手，能够记住与用户的对话，并在未来的交流中利用这些记忆提供更连贯、更有上下文感知的回答。

## 功能特点

- **持久记忆**：记住过去的对话，并在未来的交流中利用这些记忆
- **语义搜索**：使用FAISS向量数据库进行高效的语义相似度搜索
- **关系存储**：使用SQLite数据库存储对话和关系数据
- **用户友好界面**：提供直观的Web界面，支持深色/浅色模式
- **实时统计**：显示对话数量、费用等统计信息
- **控制台界面**：提供系统日志查看和基本操作功能

## 安装指南

### 前提条件

- Python 3.8+
- pip (Python包管理器)
- 互联网连接（用于API调用）

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/Persistent-memory-Neko.git
cd Persistent-memory-Neko
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置API密钥：

创建`config.json`文件，内容如下：

```json
{
  "api_key": "你的API密钥",
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "Pro/deepseek-ai/DeepSeek-V3",
  "embedding_model": "text-embedding-3-small",
  "temperature": 0.7,
  "similarity_threshold": 0.7
}
```

## 使用方法

### 启动应用

```bash
python app_flask.py
```

应用将在`http://127.0.0.1:5000`启动。

### Web界面

- **主页**：访问`http://127.0.0.1:5000`进入聊天界面
- **控制台**：访问`http://127.0.0.1:5000/console`查看系统日志和执行基本操作

### 聊天功能

1. 在输入框中输入消息
2. 按Enter键或点击发送按钮发送消息
3. AI助手将根据当前问题和相关记忆生成回复
4. 右侧面板会显示与当前问题相关的历史记忆

### 配置选项

点击界面右上角的配置按钮，可以调整以下设置：

- **API密钥**：设置API访问密钥
- **模型**：选择使用的AI模型
- **温度**：调整AI回复的创造性（0.0-1.0）
- **相似度阈值**：设置记忆匹配的相似度阈值（0.5-0.9）

### 控制台功能

在控制台界面，你可以：

- 查看系统日志
- 清除所有记忆
- 查看统计信息
- 执行基本命令

## API文档

### 聊天API

```
POST /api/chat
```

请求体：
```json
{
  "message": "用户消息",
  "api_key": "可选的API密钥",
  "model": "可选的模型名称",
  "temperature": 0.7,
  "similarity_threshold": 0.7
}
```

响应：
```json
{
  "response": "AI回复内容",
  "context_memories": [...],
  "tokens": 123,
  "cost": 0.001,
  "model": "使用的模型",
  "stats": {
    "conversation_count": 10,
    "total_cost": 0.05
  }
}
```

### 记忆API

```
GET /api/memories?query=搜索关键词
```

响应：
```json
[
  {
    "user_message": "用户消息",
    "ai_message": "AI回复",
    "timestamp": 1647123456.789,
    "similarity": 0.85
  },
  ...
]
```

### 清除记忆API

```
POST /api/clear_memory
```

响应：
```json
{
  "success": true
}
```

### 统计API

```
GET /api/stats
```

响应：
```json
{
  "conversation_count": 10,
  "total_cost": 0.05
}
```

## 项目结构

```
Persistent-memory-Neko/
├── app_flask.py          # Flask应用主文件
├── neko.py               # 原始命令行应用
├── config.json           # 配置文件
├── requirements.txt      # 依赖列表
├── faiss_index.bin       # FAISS向量索引
├── memories.json         # 记忆数据
├── neko.db               # SQLite数据库
├── logs/                 # 日志目录
├── static/               # 静态资源
│   └── js/
│       └── main.js       # 前端JavaScript
└── templates/            # HTML模板
    ├── index.html        # 主页模板
    └── console.html      # 控制台模板
```

## 故障排除

### 常见问题

1. **API密钥错误**：确保在config.json中设置了正确的API密钥
2. **数据库错误**：如果遇到数据库问题，尝试删除neko.db文件并重新启动应用
3. **内存不足**：如果应用占用过多内存，可以清除记忆或重新启动应用

### 日志查看

应用日志存储在`logs/neko_flask.log`文件中，可以通过控制台界面或直接查看文件来检查错误信息。

## 开发者信息

### 扩展功能

如果你想扩展Neko AI的功能，可以：

1. 修改`app_flask.py`添加新的API端点
2. 在`templates`目录中添加新的HTML模板
3. 在`static/js`目录中添加新的JavaScript文件

### 贡献指南

欢迎提交Pull Request或Issue来改进Neko AI。在提交代码前，请确保：

1. 代码符合PEP 8风格指南
2. 添加了适当的注释和文档
3. 所有测试都通过

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## 致谢

- OpenAI API提供AI能力
- FAISS提供高效的向量搜索
- Flask提供Web框架支持
- 所有贡献者和用户