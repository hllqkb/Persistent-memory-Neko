# Neko AI Web界面

这是Neko AI持久记忆助手的Web界面，提供了一个现代化、美观的聊天界面，让您可以通过浏览器与AI助手进行交互。

## 功能特点

- **美观的聊天界面**：使用Tailwind CSS设计的现代化界面
- **实时通信**：使用WebSocket实现实时消息传递
- **代码高亮**：支持多种编程语言的代码高亮显示
- **Markdown渲染**：支持Markdown格式的消息渲染
- **暗色/亮色主题**：支持根据系统偏好或手动切换主题
- **响应式设计**：适配桌面和移动设备
- **记忆可视化**：直观展示AI的相关记忆
- **控制台视图**：提供原始控制台输出的查看界面

## 安装与运行

1. 确保已安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 启动Web界面：
   ```bash
   python app.py
   ```

3. 浏览器将自动打开 http://127.0.0.1:5000 ，您可以开始与AI助手对话。

## 使用说明

### 主聊天界面

- 在底部输入框中输入您的问题
- 按Enter键发送消息（Shift+Enter可换行）
- 点击右上角的月亮/太阳图标切换暗色/亮色主题
- 左侧边栏显示对话历史和统计信息
- 右侧面板（在宽屏设备上）显示相关记忆

### 控制台界面

- 点击左侧边栏底部的"控制台"按钮打开控制台视图
- 控制台视图显示neko.py的原始输出
- 您可以在控制台底部输入命令，直接发送到neko.py

## 技术栈

- **后端**：Flask, Flask-SocketIO
- **前端**：Tailwind CSS, Socket.IO, Markdown-it, Highlight.js
- **AI引擎**：neko.py (持久记忆AI助手)

## 注意事项

- Web界面需要与neko.py一起运行
- 确保Neo4j数据库已正确配置并运行
- 首次运行时，系统会自动创建必要的目录结构

## 截图

![聊天界面](https://images.unsplash.com/photo-1611262588019-db6cc2032da3?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1074&q=80)

![控制台界面](https://images.unsplash.com/photo-1629654297299-c8506221ca97?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1074&q=80) 