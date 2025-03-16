// 初始化变量
let totalCost = 0;
let conversationCount = 0;
let isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
let isResizing = false;
let startHeight = 0;
let startY = 0;

// 配置
const config = {
    apiKey: localStorage.getItem('apiKey') || '',
    model: localStorage.getItem('model') || 'Pro/deepseek-ai/DeepSeek-V3',
    temperature: parseFloat(localStorage.getItem('temperature') || '0.7'),
    similarityThreshold: parseFloat(localStorage.getItem('similarityThreshold') || '0.7')
};

// 初始化Markdown解析器
const md = window.markdownit({
    highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(str, { language: lang }).value;
            } catch (__) {}
        }
        return ''; // 使用外部默认转义
    },
    breaks: true,
    linkify: true
});

// DOM元素
const messageContainer = document.getElementById('message-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const typingStatus = document.getElementById('typing-status');
const totalCostElement = document.getElementById('total-cost');
const mobileTotalCostElement = document.getElementById('mobile-total-cost');
const conversationCountElement = document.getElementById('conversation-count');
const mobileConversationCountElement = document.getElementById('mobile-conversation-count');
const themeToggle = document.getElementById('theme-toggle');
const mobileThemeToggle = document.getElementById('mobile-theme-toggle');
const sidebarToggle = document.getElementById('sidebar-toggle');
const mobileSidebar = document.getElementById('mobile-sidebar');
const closeSidebar = document.getElementById('close-sidebar');
const memoriesContainer = document.getElementById('memories-container');
const conversationHistory = document.getElementById('conversation-history');
const mobileConversationHistory = document.getElementById('mobile-conversation-history');
const consoleContainer = document.getElementById('console-container');
const consoleResizer = document.getElementById('console-resizer');
const configButton = document.getElementById('config-button');
const configPanel = document.getElementById('config-panel');
const saveConfigButton = document.getElementById('save-config');
const clearConsoleButton = document.getElementById('clear-console');
const clearMemoryButton = document.getElementById('clear-memory');
const apiKeyInput = document.getElementById('api-key');
const modelSelect = document.getElementById('model-select');
const temperatureSlider = document.getElementById('temperature');
const openSidebarButton = document.getElementById('sidebar-toggle');
const consoleOutput = document.querySelector('.console-output');
const toggleConsole = document.getElementById('toggle-console');
const mobileConfigButton = document.getElementById('mobile-config-button');
const similaritySlider = document.getElementById('similarity-threshold');

// 初始化函数
function init() {
    // 设置主题
    if (isDarkMode) {
        document.documentElement.classList.add('dark');
    }

    // 设置事件监听器
    setupEventListeners();

    // 自动调整文本区域高度
    autoResizeTextarea();
    
    // 初始化控制台大小调整
    initConsoleResizer();
    
    // 加载配置
    loadConfig();
    
    // 获取统计信息
    fetchStats();
    
    // 添加欢迎消息
    if (messageContainer && messageContainer.children.length <= 1) {
        const welcomeMessage = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <div class="h-10 w-10 rounded-full bg-primary-500 flex items-center justify-center text-white">
                        <i class="fas fa-robot"></i>
                    </div>
                </div>
                <div class="ml-3 bg-white dark:bg-gray-800 rounded-lg px-4 py-3 max-w-3xl shadow-sm">
                    <div class="text-sm text-gray-900 dark:text-white">
                        <p>👋 你好！我是Neko AI，一个具有持久记忆能力的AI助手。</p>
                        <p class="mt-2">我可以记住我们之间的对话内容，并在未来的交流中利用这些记忆来提供更连贯、更有上下文感知的回答。</p>
                        <p class="mt-2">有什么我可以帮助你的吗？</p>
                    </div>
                    <div class="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        刚刚
                    </div>
                </div>
            </div>
        `;
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'mb-4';
        welcomeDiv.innerHTML = welcomeMessage;
        messageContainer.appendChild(welcomeDiv);
    }
}

// 初始化控制台大小调整
function initConsoleResizer() {
    if (!consoleResizer || !consoleContainer) return;
    
    consoleResizer.addEventListener('mousedown', function(e) {
        isResizing = true;
        startHeight = consoleContainer.offsetHeight;
        startY = e.clientY;
        
        document.addEventListener('mousemove', resizeConsole);
        document.addEventListener('mouseup', stopResizing);
        
        // 防止选中文本
        e.preventDefault();
    });
    
    // 控制台切换按钮
    const toggleConsole = document.getElementById('toggle-console');
    if (toggleConsole) {
        toggleConsole.addEventListener('click', function() {
            if (consoleContainer.style.height === '0px') {
                consoleContainer.style.height = '200px';
                this.innerHTML = '<i class="fas fa-chevron-down"></i>';
            } else {
                consoleContainer.style.height = '0px';
                this.innerHTML = '<i class="fas fa-chevron-up"></i>';
            }
        });
    }
}

// 调整控制台大小
function resizeConsole(e) {
    if (!isResizing) return;
    
    const delta = startY - e.clientY;
    const newHeight = Math.max(50, Math.min(500, startHeight + delta));
    consoleContainer.style.height = newHeight + 'px';
}

// 停止调整大小
function stopResizing() {
    isResizing = false;
    document.removeEventListener('mousemove', resizeConsole);
    document.removeEventListener('mouseup', stopResizing);
}

// 添加控制台输出
function addConsoleOutput(text) {
    if (!consoleContainer) return;
    
    const consoleOutput = consoleContainer.querySelector('.console-output');
    if (!consoleOutput) return;
    
    const line = document.createElement('div');
    line.className = 'console-line';
    
    // 处理ANSI颜色代码
    text = processAnsiCodes(text);
    
    line.innerHTML = text;
    consoleOutput.appendChild(line);
    
    // 滚动到底部
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

// 处理ANSI颜色代码
function processAnsiCodes(text) {
    // 替换一些常见的颜色代码
    text = text.replace(/\u001b\[31m/g, '<span class="ansi-red">');
    text = text.replace(/\u001b\[32m/g, '<span class="ansi-green">');
    text = text.replace(/\u001b\[33m/g, '<span class="ansi-yellow">');
    text = text.replace(/\u001b\[34m/g, '<span class="ansi-blue">');
    text = text.replace(/\u001b\[35m/g, '<span class="ansi-magenta">');
    text = text.replace(/\u001b\[36m/g, '<span class="ansi-cyan">');
    text = text.replace(/\u001b\[37m/g, '<span class="ansi-white">');
    text = text.replace(/\u001b\[0m/g, '</span>');
    
    return text;
}

// 清空控制台
function clearConsole() {
    if (!consoleContainer) return;
    
    const consoleOutput = consoleContainer.querySelector('.console-output');
    if (consoleOutput) {
        consoleOutput.innerHTML = '';
    }
}

// 设置事件监听器
function setupEventListeners() {
    if (sendButton) {
        sendButton.addEventListener('click', function() {
            const message = messageInput.value.trim();
            if (message) {
                sendMessage(message);
            }
        });
    }
    
    if (messageInput) {
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const message = messageInput.value.trim();
                if (message) {
                    sendMessage(message);
                }
            }
        });
    }
    
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleDarkMode);
    }
    
    if (mobileThemeToggle) {
        mobileThemeToggle.addEventListener('click', toggleDarkMode);
    }
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            mobileSidebar.classList.remove('hidden');
        });
    }
    
    if (closeSidebar) {
        closeSidebar.addEventListener('click', () => {
            mobileSidebar.classList.add('hidden');
        });
    }
    
    if (configButton) {
        configButton.addEventListener('click', () => {
            configPanel.classList.remove('hidden');
        });
    }
    
    if (saveConfigButton) {
        saveConfigButton.addEventListener('click', saveConfig);
    }
    
    if (clearConsoleButton) {
        clearConsoleButton.addEventListener('click', clearConsole);
    }
    
    if (clearMemoryButton) {
        clearMemoryButton.addEventListener('click', clearMemory);
    }
    
    // 添加错误处理
    window.addEventListener('error', function(e) {
        console.error('全局错误:', e.message);
        addConsoleOutput(`[错误] ${e.message}`);
    });
    
    // 添加未处理的Promise拒绝处理
    window.addEventListener('unhandledrejection', function(e) {
        console.error('未处理的Promise拒绝:', e.reason);
        addConsoleOutput(`[Promise错误] ${e.reason}`);
    });
}

// 加载配置
function loadConfig() {
    apiKeyInput.value = config.apiKey;
    modelSelect.value = config.model;
    temperatureSlider.value = config.temperature;
    document.getElementById('temperature-value').textContent = config.temperature;
    similaritySlider.value = config.similarityThreshold;
    document.getElementById('similarity-value').textContent = config.similarityThreshold;
}

// 保存配置
function saveConfig() {
    config.apiKey = apiKeyInput.value;
    config.model = modelSelect.value;
    config.temperature = temperatureSlider.value;
    config.similarityThreshold = similaritySlider.value;
    
    localStorage.setItem('apiKey', config.apiKey);
    localStorage.setItem('model', config.model);
    localStorage.setItem('temperature', config.temperature);
    localStorage.setItem('similarityThreshold', config.similarityThreshold);
    
    configPanel.classList.add('hidden');
    
    // 更新统计信息
    fetchStats();
    
    // 显示提示
    showToast('配置已保存');
}

// 获取统计信息
function fetchStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            if (data.memory_count !== undefined) {
                conversationCount = data.memory_count;
                if (conversationCountElement) {
                    conversationCountElement.textContent = data.memory_count;
                }
                if (mobileConversationCountElement) {
                    mobileConversationCountElement.textContent = data.memory_count;
                }
            }
            
            if (data.total_cost !== undefined) {
                totalCost = data.total_cost;
                const formattedCost = '¥' + data.total_cost.toFixed(4);
                if (totalCostElement) {
                    totalCostElement.textContent = formattedCost;
                }
                if (mobileTotalCostElement) {
                    mobileTotalCostElement.textContent = formattedCost;
                }
            }
        })
        .catch(error => console.error('获取统计信息失败:', error));
}

// 显示提示消息
function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded-md shadow-lg transition-opacity duration-300';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('opacity-0');
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

// 发送消息
async function sendMessage(message) {
    if (!message || !message.trim()) return;
    
    // 清空输入框
    messageInput.value = '';
    
    // 添加用户消息到聊天界面
    addMessage('user', message);
    
    // 显示思考状态
    if (typingStatus) {
        typingStatus.classList.remove('hidden');
    }
    
    // 添加AI正在输入的提示
    const aiMessageElement = addMessage('assistant', '<div class="typing-indicator">AI思考中</div>');
    
    try {
        // 发送请求到API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                api_key: config.apiKey,
                model: config.model,
                temperature: parseFloat(config.temperature),
                similarity_threshold: parseFloat(config.similarityThreshold)
            })
        });
        
        if (!response.ok) {
            throw new Error('API请求失败');
        }
        
        const data = await response.json();
        
        // 隐藏思考状态
        if (typingStatus) {
            typingStatus.classList.add('hidden');
        }
        
        // 更新AI消息
        aiMessageElement.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                        <i class="fas fa-robot text-primary-600"></i>
                    </div>
                </div>
                <div class="ml-3">
                    <div class="text-sm text-gray-700 dark:text-gray-300 prose dark:prose-dark max-w-none">
                        ${md.render(data.response)}
                    </div>
                    <div class="mt-1 text-xs text-gray-500">
                        输入: ${data.stats.input_tokens} tokens | 
                        输出: ${data.stats.output_tokens} tokens | 
                        费用: ¥${data.stats.cost.toFixed(6)} | 
                        用时: ${(data.stats.time).toFixed(2)}s
                    </div>
                </div>
            </div>
        `;
        
        // 更新统计信息
        fetchStats();
        
        // 添加到对话历史
        addToHistory(message, data.response);
        
        // 高亮代码块
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
        });
        
        // 滚动到底部
        scrollToBottom();
    } catch (error) {
        console.error('发送消息失败:', error);
        
        // 隐藏思考状态
        if (typingStatus) {
            typingStatus.classList.add('hidden');
        }
        
        // 显示错误消息
        aiMessageElement.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-red-100 flex items-center justify-center">
                        <i class="fas fa-exclamation-triangle text-red-600"></i>
                    </div>
                </div>
                <div class="ml-3">
                    <div class="text-sm text-red-600">
                        发生错误: ${error.message}
                    </div>
                </div>
            </div>
        `;
    }
}

// 添加消息到聊天界面
function addMessage(role, content) {
    const messageElement = document.createElement('div');
    messageElement.className = 'mb-4';
    
    if (role === 'user') {
        messageElement.innerHTML = `
            <div class="flex items-start justify-end">
                <div class="mr-3">
                    <div class="text-sm bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 rounded-lg py-2 px-4 max-w-md">
                        ${content}
                    </div>
                </div>
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-primary-500 flex items-center justify-center">
                        <i class="fas fa-user text-white"></i>
                    </div>
                </div>
            </div>
        `;
    } else {
        messageElement.innerHTML = content;
    }
    
    if (messageContainer) {
        messageContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    return messageElement;
}

// 添加到对话历史
function addToHistory(userMessage, aiResponse) {
    const historyItem = document.createElement('a');
    historyItem.href = '#';
    historyItem.className = 'block px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md';
    historyItem.innerHTML = `
        <div class="truncate font-medium">${userMessage.substring(0, 30)}${userMessage.length > 30 ? '...' : ''}</div>
        <div class="truncate text-xs text-gray-500">${new Date().toLocaleString()}</div>
    `;
    
    historyItem.addEventListener('click', function(e) {
        e.preventDefault();
        messageInput.value = userMessage;
        messageInput.focus();
    });
    
    // 添加到桌面和移动端历史记录
    if (conversationHistory) {
        conversationHistory.insertBefore(historyItem.cloneNode(true), conversationHistory.firstChild);
    }
    
    if (mobileConversationHistory) {
        const mobileItem = historyItem.cloneNode(true);
        mobileItem.addEventListener('click', function(e) {
            e.preventDefault();
            messageInput.value = userMessage;
            messageInput.focus();
            mobileSidebar.classList.add('hidden');
        });
        mobileConversationHistory.insertBefore(mobileItem, mobileConversationHistory.firstChild);
    }
}

// 清除记忆
async function clearMemory() {
    if (!confirm('确定要清除所有记忆吗？此操作不可撤销。')) {
        return;
    }
    
    try {
        const response = await fetch('/api/clear_memory', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('清除记忆失败');
        }
        
        const data = await response.json();
        alert(data.message);
        
        // 清空历史记录UI
        if (conversationHistory) conversationHistory.innerHTML = '';
        if (mobileConversationHistory) mobileConversationHistory.innerHTML = '';
        
        // 更新统计信息
        fetchStats();
    } catch (error) {
        console.error('清除记忆失败:', error);
        alert('清除记忆失败: ' + error.message);
    }
}

// 切换暗色模式
function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    if (isDarkMode) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
}

// 自动调整文本区域高度
function autoResizeTextarea() {
    if (!messageInput) return;
    
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// 滚动到底部
function scrollToBottom() {
    if (messageContainer) {
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
}

// 格式化时间
function formatTime(date) {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
}

// 初始化
document.addEventListener('DOMContentLoaded', init); 