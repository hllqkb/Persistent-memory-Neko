# Neko AI Flask 前端

这是一个基于Flask的前端界面，用于与Neko AI持久记忆助手进行交互。该前端提供了一个Web界面，允许用户通过浏览器与AI助手对话，并查看对话历史和统计信息。

## 功能特点

- 美观的Web界面，支持桌面和移动设备
- 实时对话与AI助手
- 查看对话历史和相关记忆
- 显示token使用和费用统计
- 支持Markdown和代码高亮
- 配置API密钥和模型参数
- 清除记忆功能

## 安装

1. 确保已安装所有依赖：

```bash
pip install -r requirements.txt
```

2. 配置API密钥：

在`config.json`文件中设置您的API密钥和其他配置：

```json
{
  "api_key": "您的API密钥",
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "Pro/deepseek-ai/DeepSeek-V3"
}
```

## 使用方法

1. 启动Flask应用：

```bash
python app.py
```

2. 在浏览器中访问：

```
http://localhost:5000
```

3. 开始与AI助手对话！

## API端点

### 1. 聊天API

- **URL**: `/api/chat`
- **方法**: POST
- **请求体**:
  ```json
  {
    "message": "用户消息"
  }
  ```
- **响应**:
  ```json
  {
    "response": "AI助手的回复",
    "stats": {
      "input_tokens": 123,
      "output_tokens": 456,
      "cost": 0.00123,
      "total_cost": 0.01234,
      "time": 1.23
    }
  }
  ```

### 2. 获取相似记忆API

- **URL**: `/api/memories`
- **方法**: POST
- **请求体**:
  ```json
  {
    "query": "搜索查询",
    "k": 5
  }
  ```
- **响应**:
  ```json
  {
    "memories": [
      {
        "user_message": "用户消息",
        "ai_response": "AI回复",
        "timestamp": "2023-01-01 12:00:00.000000",
        "similarity": 0.95
      }
    ]
  }
  ```

### 3. 清除记忆API

- **URL**: `/api/clear_memory`
- **方法**: POST
- **响应**:
  ```json
  {
    "success": true,
    "message": "所有记忆已清除"
  }
  ```

### 4. 获取统计信息API

- **URL**: `/api/stats`
- **方法**: GET
- **响应**:
  ```json
  {
    "memory_count": 123,
    "total_cost": 0.01234
  }
  ```

## 文件结构

- `app.py` - Flask应用主文件
- `neko.py` - 核心AI功能
- `templates/` - HTML模板
  - `index.html` - 主聊天界面
  - `console.html` - 控制台界面
- `static/` - 静态资源
  - `js/main.js` - 前端JavaScript
  - `css/` - 样式表
  - `img/` - 图片资源

## 注意事项

- 确保Neo4j数据库已启动并配置正确
- 确保API密钥有效
- 对于生产环境，建议使用Gunicorn或uWSGI作为WSGI服务器

## 故障排除

如果遇到问题，请检查：

1. 日志文件 (`logs/neko.log`)
2. 确保所有依赖已正确安装
3. 确保Neo4j数据库连接正常
4. 确保API密钥有效

## 许可证

与原Neko AI项目相同的许可证 