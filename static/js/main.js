// 初始化变量
let totalCost = 0;
let conversationCount = 0;
let isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
let isResizing = false;
let startHeight = 0;
let startY = 0;
let currentConversationTimestamp;
let currentGroupId = null; // 当前对话组ID

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
    
    // 获取对话历史
    fetchChatHistory();
    
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
function sendMessage() {
    if (!messageInput) return;
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // 清空输入框
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // 添加用户消息到界面
    addMessage('user', message);
    
    // 添加AI思考中的消息
    const aiMessageElement = document.createElement('div');
    aiMessageElement.className = 'mb-4';
    aiMessageElement.innerHTML = `
        <div class="flex items-start">
            <div class="flex-shrink-0">
                <div class="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                    <i class="fas fa-robot text-primary-600"></i>
                </div>
            </div>
            <div class="ml-3 bg-white dark:bg-gray-800 rounded-lg px-4 py-3 max-w-3xl shadow-sm">
                <div class="text-sm text-gray-900 dark:text-white">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    if (messageContainer) {
        messageContainer.appendChild(aiMessageElement);
        scrollToBottom();
    }
    
    // 准备请求数据
    const requestData = {
        message: message
    };
    
    // 如果是继续对话组对话，添加对话组ID
    if (currentGroupId) {
        requestData.group_id = currentGroupId;
    }
    // 如果是继续特定对话，添加时间戳信息
    else if (currentConversationTimestamp) {
        requestData.conversation_timestamp = currentConversationTimestamp;
        // 更新当前时间戳为null，表示这是一个新的对话轮次
        currentConversationTimestamp = null;
    }
    
    // 发送请求到服务器
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('发送消息失败: ' + response.statusText);
        }
        
        // 获取响应流
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullResponse = '';
        let statsData = null;
        
        // 处理响应流
        function processStream({ done, value }) {
            if (done) {
                // 处理可能的剩余数据
                if (buffer) {
                    try {
                        const lines = buffer.split('\n\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = JSON.parse(line.substring(6));
                                if (data.done) {
                                    statsData = data.stats;
                                } else if (data.content) {
                                    fullResponse += data.content;
                                }
                            }
                        }
                    } catch (e) {
                        console.error('解析剩余数据时出错:', e);
                    }
                }
                
                // 更新AI消息元素
                if (aiMessageElement) {
                    aiMessageElement.innerHTML = `
                        <div class="flex items-start">
                            <div class="flex-shrink-0">
                                <div class="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                                    <i class="fas fa-robot text-primary-600"></i>
                                </div>
                            </div>
                            <div class="ml-3">
                                <div class="text-sm text-gray-700 dark:text-gray-300 prose dark:prose-dark max-w-none">
                                    ${md.render(fullResponse)}
                                </div>
                                ${statsData ? `
                                <div class="mt-1 text-xs text-gray-500">
                                    tokens: ${statsData.input_tokens + statsData.output_tokens} | 费用: ¥${statsData.cost.toFixed(6)}
                                </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
                    
                    // 高亮代码块
                    aiMessageElement.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightBlock(block);
                    });
                    
                    scrollToBottom();
                }
                
                // 如果返回了对话组ID，更新当前对话组ID
                if (statsData && statsData.group_id) {
                    currentGroupId = statsData.group_id;
                }
                
                // 添加到对话历史
                if (fullResponse) {
                    addToHistory(message, fullResponse);
                }
                
                // 更新统计信息
                if (statsData) {
                    updateStats(statsData);
                }
                
                return;
            }
            
            // 处理数据块
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;
            
            // 处理完整的SSE消息
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        
                        if (data.error) {
                            // 处理错误
                            aiMessageElement.innerHTML = `
                                <div class="flex items-start">
                                    <div class="flex-shrink-0">
                                        <div class="h-8 w-8 rounded-full bg-red-100 flex items-center justify-center">
                                            <i class="fas fa-exclamation-triangle text-red-600"></i>
                                        </div>
                                    </div>
                                    <div class="ml-3">
                                        <div class="text-sm text-red-600">
                                            错误: ${data.error}
                                        </div>
                                    </div>
                                </div>
                            `;
                            return;
                        } else if (data.done) {
                            // 保存统计信息
                            statsData = data.stats;
                        } else if (data.content) {
                            // 添加内容到完整响应
                            fullResponse += data.content;
                            
                            // 更新AI消息元素
                            aiMessageElement.innerHTML = `
                                <div class="flex items-start">
                                    <div class="flex-shrink-0">
                                        <div class="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                                            <i class="fas fa-robot text-primary-600"></i>
                                        </div>
                                    </div>
                                    <div class="ml-3">
                                        <div class="text-sm text-gray-700 dark:text-gray-300 prose dark:prose-dark max-w-none">
                                            ${md.render(fullResponse)}
                                        </div>
                                    </div>
                                </div>
                            `;
                            
                            // 高亮代码块
                            aiMessageElement.querySelectorAll('pre code').forEach((block) => {
                                hljs.highlightBlock(block);
                            });
                            
                            scrollToBottom();
                        }
                    } catch (e) {
                        console.error('解析SSE消息时出错:', e, line);
                    }
                }
            }
            
            // 继续读取流
            return reader.read().then(processStream);
        }
        
        // 开始处理流
        return reader.read().then(processStream);
    })
    .catch(error => {
        console.error('发送消息失败:', error);
        
        // 更新AI消息元素显示错误
        aiMessageElement.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-red-100 flex items-center justify-center">
                        <i class="fas fa-exclamation-triangle text-red-600"></i>
                    </div>
                </div>
                <div class="ml-3">
                    <div class="text-sm text-red-600">
                        发送消息失败: ${error.message}
                    </div>
                </div>
            </div>
        `;
        
        addConsoleOutput(`[错误] 发送消息失败: ${error.message}`);
    });
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

// 获取对话历史
function fetchChatHistory() {
    fetch('/api/conversation_groups?page=1&per_page=10')
        .then(response => {
            if (!response.ok) {
                throw new Error('获取对话组列表失败: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data.groups && data.groups.length > 0) {
                // 清空现有历史记录
                if (conversationHistory) {
                    conversationHistory.innerHTML = '';
                }
                if (mobileConversationHistory) {
                    mobileConversationHistory.innerHTML = '';
                }
                
                // 添加新建对话按钮
                addNewChatButton();
                
                // 添加新的历史记录
                data.groups.forEach(group => {
                    const historyItem = document.createElement('a');
                    historyItem.href = '#';
                    historyItem.className = 'block px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md';
                    
                    // 确保标题存在
                    const title = group.title || '无标题对话';
                    const groupId = group.id;
                    
                    historyItem.innerHTML = `
                        <div class="truncate font-medium">${title}</div>
                        <div class="truncate text-xs text-gray-500">${group.formatted_time || new Date(group.created_at * 1000).toLocaleString()} (${group.message_count}条消息)</div>
                    `;
                    
                    // 直接使用函数声明而不是匿名函数
                    historyItem.onclick = function(e) {
                        e.preventDefault();
                        // 加载对话组内容
                        loadConversationGroup(groupId);
                        console.log('对话组点击: ' + title + ', ID: ' + groupId);
                    };
                    
                    if (conversationHistory) {
                        conversationHistory.appendChild(historyItem);
                    }
                    
                    if (mobileConversationHistory) {
                        const mobileItem = historyItem.cloneNode(true);
                        // 直接使用函数声明
                        mobileItem.onclick = function(e) {
                            e.preventDefault();
                            // 加载对话组内容
                            loadConversationGroup(groupId);
                            mobileSidebar.classList.add('hidden');
                            console.log('移动端对话组点击: ' + title + ', ID: ' + groupId);
                        };
                        mobileConversationHistory.appendChild(mobileItem);
                    }
                });
            } else {
                // 如果没有对话历史，也添加新建对话按钮
                addNewChatButton();
                
                if (data.error) {
                    console.error('获取对话组列表错误:', data.error);
                    addConsoleOutput(`[错误] 获取对话组列表错误: ${data.error}`);
                }
            }
        })
        .catch(error => {
            console.error('获取对话组列表失败:', error);
            addConsoleOutput(`[错误] 获取对话组列表失败: ${error.message}`);
            
            // 出错时也添加新建对话按钮
            addNewChatButton();
        });
}

// 加载完整对话历史
function loadFullConversation(timestamp) {
    // 保存当前时间戳到全局变量，用于继续对话
    currentConversationTimestamp = timestamp;
    
    // 显示加载状态
    if (messageContainer) {
        messageContainer.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center">
                    <div class="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500"></div>
                    <p class="mt-2 text-gray-600 dark:text-gray-400">正在加载对话...</p>
                </div>
            </div>
        `;
    }
    
    // 获取特定对话及其上下文
    fetch(`/api/full_conversation?timestamp=${timestamp}&before=-1&after=-1`)
        .then(response => {
            if (!response.ok) {
                throw new Error('获取对话失败: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data.conversations && data.conversations.length > 0) {
                // 清空消息容器
                if (messageContainer) {
                    messageContainer.innerHTML = '';
                    
                    // 添加对话标题
                    const titleElement = document.createElement('div');
                    titleElement.className = 'text-center py-4 border-b border-gray-200 dark:border-gray-700 mb-4';
                    titleElement.innerHTML = `
                        <h2 class="text-lg font-medium text-gray-900 dark:text-white">历史对话</h2>
                        <p class="text-sm text-gray-500 dark:text-gray-400">正在继续该对话</p>
                    `;
                    messageContainer.appendChild(titleElement);
                    
                    // 显示所有对话
                    data.conversations.forEach(conv => {
                        // 高亮当前选中的对话
                        const isCurrentConversation = Math.abs(conv.timestamp - timestamp) < 0.001;
                        
                        // 添加用户消息
                        const userMessageElement = document.createElement('div');
                        userMessageElement.className = 'mb-4';
                        userMessageElement.innerHTML = `
                            <div class="flex items-start justify-end">
                                <div class="mr-3">
                                    <div class="text-sm bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 rounded-lg py-2 px-4 max-w-md ${isCurrentConversation ? 'border-2 border-primary-500' : ''}">
                                        ${conv.user_message}
                                    </div>
                                    <div class="mt-1 text-xs text-right text-gray-500 dark:text-gray-400">
                                        ${conv.formatted_time}
                                    </div>
                                </div>
                                <div class="flex-shrink-0">
                                    <div class="h-8 w-8 rounded-full bg-primary-500 flex items-center justify-center">
                                        <i class="fas fa-user text-white"></i>
                                    </div>
                                </div>
                            </div>
                        `;
                        messageContainer.appendChild(userMessageElement);
                        
                        // 添加AI响应
                        const aiMessageElement = document.createElement('div');
                        aiMessageElement.className = 'mb-4';
                        aiMessageElement.innerHTML = `
                            <div class="flex items-start">
                                <div class="flex-shrink-0">
                                    <div class="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                                        <i class="fas fa-robot text-primary-600"></i>
                                    </div>
                                </div>
                                <div class="ml-3">
                                    <div class="text-sm text-gray-700 dark:text-gray-300 prose dark:prose-dark max-w-none ${isCurrentConversation ? 'border-2 border-primary-500 p-2 rounded-lg' : ''}">
                                        ${md.render(conv.ai_message)}
                                    </div>
                                    <div class="mt-1 text-xs text-gray-500">
                                        tokens: ${conv.tokens} | 费用: ¥${conv.cost.toFixed(6)}
                                    </div>
                                </div>
                            </div>
                        `;
                        messageContainer.appendChild(aiMessageElement);
                    });
                    
                    // 添加"继续对话"提示
                    const continueElement = document.createElement('div');
                    continueElement.className = 'text-center py-4 mt-4 border-t border-gray-200 dark:border-gray-700';
                    continueElement.innerHTML = `
                        <p class="text-sm text-gray-600 dark:text-gray-400">在下方输入框中输入消息，继续这个对话</p>
                    `;
                    messageContainer.appendChild(continueElement);
                    
                    // 高亮代码块
                    messageContainer.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightBlock(block);
                    });
                    
                    // 滚动到底部
                    messageContainer.scrollTop = messageContainer.scrollHeight;
                    
                    // 聚焦输入框
                    if (messageInput) {
                        messageInput.focus();
                    }
                }
            } else {
                // 没有找到对话
                if (messageContainer) {
                    messageContainer.innerHTML = `
                        <div class="flex items-center justify-center h-full">
                            <div class="text-center">
                                <i class="fas fa-exclamation-circle text-3xl text-gray-400 mb-2"></i>
                                <p class="text-gray-600 dark:text-gray-400">未找到对话</p>
                            </div>
                        </div>
                    `;
                }
                
                if (data.error) {
                    console.error('获取对话错误:', data.error);
                    addConsoleOutput(`[错误] 获取对话错误: ${data.error}`);
                }
            }
        })
        .catch(error => {
            console.error('获取对话失败:', error);
            addConsoleOutput(`[错误] 获取对话失败: ${error.message}`);
            
            // 显示错误信息
            if (messageContainer) {
                messageContainer.innerHTML = `
                    <div class="flex items-center justify-center h-full">
                        <div class="text-center">
                            <i class="fas fa-exclamation-triangle text-3xl text-red-500 mb-2"></i>
                            <p class="text-red-600">加载对话失败: ${error.message}</p>
                        </div>
                    </div>
                `;
            }
        });
}

// 加载对话上下文 (保留此函数以兼容性，但实际上不再需要单独的上下文加载)
function loadConversationContext(timestamp) {
    loadFullConversation(timestamp);
}

// 添加新建对话按钮
function addNewChatButton() {
    // 创建新建对话按钮
    const newChatButton = document.createElement('a');
    newChatButton.href = '#';
    newChatButton.className = 'block px-3 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md text-center mb-2';
    newChatButton.innerHTML = '<i class="fas fa-plus mr-2"></i>新建对话组';
    
    // 添加点击事件
    newChatButton.onclick = function(e) {
        e.preventDefault();
        
        // 创建新的对话组
        const title = prompt('请输入对话组标题:', '新对话组 ' + new Date().toLocaleString());
        if (title) {
            createNewConversationGroup(title);
        }
        
        console.log('新建对话组');
        
        // 关闭移动端侧边栏
        if (mobileSidebar) {
            mobileSidebar.classList.add('hidden');
        }
    };
    
    // 添加到桌面历史记录顶部
    if (conversationHistory) {
        conversationHistory.insertBefore(newChatButton, conversationHistory.firstChild);
    }
    
    // 添加到移动端历史记录顶部
    if (mobileConversationHistory) {
        const mobileNewChatButton = newChatButton.cloneNode(true);
        mobileNewChatButton.onclick = newChatButton.onclick;
        mobileConversationHistory.insertBefore(mobileNewChatButton, mobileConversationHistory.firstChild);
    }
}

// 创建新的对话组
function createNewConversationGroup(title) {
    fetch('/api/conversation_groups', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            title: title
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('创建对话组失败: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.success && data.group) {
            // 更新当前对话组ID
            currentGroupId = data.group.id;
            
            // 清空消息容器，准备开始新对话
            if (messageContainer) {
                messageContainer.innerHTML = '';
                
                // 添加对话组标题
                const titleElement = document.createElement('div');
                titleElement.className = 'text-center py-4 border-b border-gray-200 dark:border-gray-700 mb-4';
                titleElement.innerHTML = `
                    <h2 class="text-lg font-medium text-gray-900 dark:text-white">${data.group.title}</h2>
                    <p class="text-sm text-gray-500 dark:text-gray-400">新对话组</p>
                    <div class="mt-2 flex justify-center space-x-2">
                        <button id="rename-group" class="px-3 py-1 text-xs text-primary-600 border border-primary-600 rounded-md hover:bg-primary-50 dark:hover:bg-primary-900">
                            重命名对话组
                        </button>
                        <button id="delete-group" class="px-3 py-1 text-xs text-red-600 border border-red-600 rounded-md hover:bg-red-50 dark:hover:bg-red-900">
                            删除对话组
                        </button>
                    </div>
                `;
                messageContainer.appendChild(titleElement);
                
                // 添加重命名对话组的事件
                const renameGroupButton = document.getElementById('rename-group');
                if (renameGroupButton) {
                    renameGroupButton.onclick = function() {
                        const newTitle = prompt('请输入新的对话组标题:', data.group.title);
                        if (newTitle && newTitle.trim()) {
                            updateConversationGroup(data.group.id, newTitle.trim());
                        }
                    };
                }
                
                // 添加删除对话组的事件
                const deleteGroupButton = document.getElementById('delete-group');
                if (deleteGroupButton) {
                    deleteGroupButton.onclick = function() {
                        if (confirm('确定要删除这个对话组吗？此操作不可撤销。')) {
                            deleteConversationGroup(data.group.id);
                        }
                    };
                }
                
                // 添加欢迎消息
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
            
            // 更新侧边栏
            fetchChatHistory();
            
            // 聚焦输入框
            if (messageInput) {
                messageInput.focus();
            }
            
            showToast('已创建新对话组');
        } else {
            throw new Error(data.error || '创建对话组失败');
        }
    })
    .catch(error => {
        console.error('创建对话组失败:', error);
        addConsoleOutput(`[错误] 创建对话组失败: ${error.message}`);
        showToast('创建对话组失败: ' + error.message, 'error');
    });
}

// 添加到对话历史
function addToHistory(userMessage, aiResponse) {
    // 创建一个新的历史项
    const historyItem = document.createElement('a');
    historyItem.href = '#';
    historyItem.className = 'block px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md';
    
    // 获取当前时间戳
    const currentTime = new Date();
    const timestamp = currentTime.getTime() / 1000; // 转换为秒
    
    historyItem.innerHTML = `
        <div class="truncate font-medium">${userMessage.substring(0, 30)}${userMessage.length > 30 ? '...' : ''}</div>
        <div class="truncate text-xs text-gray-500">${currentTime.toLocaleString()}</div>
    `;
    
    // 使用onclick而不是addEventListener
    historyItem.onclick = function(e) {
        e.preventDefault();
        // 加载完整对话历史
        loadFullConversation(timestamp);
        console.log('新历史项点击: ' + userMessage + ', 时间戳: ' + timestamp);
    };
    
    // 添加到桌面历史记录
    if (conversationHistory) {
        // 检查是否已有新建对话按钮
        const newChatButton = conversationHistory.querySelector('a.bg-primary-600');
        if (newChatButton) {
            // 在新建对话按钮后插入
            conversationHistory.insertBefore(historyItem, newChatButton.nextSibling);
        } else {
            // 没有新建对话按钮，先添加一个
            addNewChatButton();
            // 然后在按钮后插入
            conversationHistory.insertBefore(historyItem, conversationHistory.firstChild.nextSibling);
        }
    }
    
    // 添加到移动端历史记录
    if (mobileConversationHistory) {
        const mobileItem = historyItem.cloneNode(true);
        mobileItem.onclick = function(e) {
            e.preventDefault();
            // 加载完整对话历史
            loadFullConversation(timestamp);
            mobileSidebar.classList.add('hidden');
            console.log('移动端新历史项点击: ' + userMessage + ', 时间戳: ' + timestamp);
        };
        
        // 检查是否已有新建对话按钮
        const mobileNewChatButton = mobileConversationHistory.querySelector('a.bg-primary-600');
        if (mobileNewChatButton) {
            // 在新建对话按钮后插入
            mobileConversationHistory.insertBefore(mobileItem, mobileNewChatButton.nextSibling);
        } else {
            // 没有新建对话按钮，先添加一个
            addNewChatButton();
            // 然后在按钮后插入
            mobileConversationHistory.insertBefore(mobileItem, mobileConversationHistory.firstChild.nextSibling);
        }
    }
    
    // 延迟获取历史对话记录，避免立即刷新导致的闪烁
    setTimeout(fetchChatHistory, 1000);
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

// 加载单轮对话（保留原函数，以防需要）
function loadConversation(userMessage, aiMessage) {
    // 清空消息容器
    if (messageContainer) {
        messageContainer.innerHTML = '';
        
        // 添加用户消息
        const userMessageElement = document.createElement('div');
        userMessageElement.className = 'mb-4';
        userMessageElement.innerHTML = `
            <div class="flex items-start justify-end">
                <div class="mr-3">
                    <div class="text-sm bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 rounded-lg py-2 px-4 max-w-md">
                        ${userMessage}
                    </div>
                </div>
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-primary-500 flex items-center justify-center">
                        <i class="fas fa-user text-white"></i>
                    </div>
                </div>
            </div>
        `;
        messageContainer.appendChild(userMessageElement);
        
        // 添加AI响应
        const aiMessageElement = document.createElement('div');
        aiMessageElement.className = 'mb-4';
        aiMessageElement.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                        <i class="fas fa-robot text-primary-600"></i>
                    </div>
                </div>
                <div class="ml-3">
                    <div class="text-sm text-gray-700 dark:text-gray-300 prose dark:prose-dark max-w-none">
                        ${md.render(aiMessage)}
                    </div>
                </div>
            </div>
        `;
        messageContainer.appendChild(aiMessageElement);
        
        // 高亮代码块
        messageContainer.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
        });
        
        // 滚动到顶部
        messageContainer.scrollTop = 0;
    }
}

// 加载对话组内容
function loadConversationGroup(groupId) {
    // 保存当前对话组ID
    currentGroupId = groupId;
    
    // 显示加载状态
    if (messageContainer) {
        messageContainer.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center">
                    <div class="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-500"></div>
                    <p class="mt-2 text-gray-600 dark:text-gray-400">正在加载对话组...</p>
                </div>
            </div>
        `;
    }
    
    // 获取对话组内容
    fetch(`/api/conversation_group/${groupId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('获取对话组内容失败: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data.conversations && data.conversations.length > 0) {
                // 清空消息容器
                if (messageContainer) {
                    messageContainer.innerHTML = '';
                    
                    // 添加对话组标题
                    const titleElement = document.createElement('div');
                    titleElement.className = 'text-center py-4 border-b border-gray-200 dark:border-gray-700 mb-4';
                    titleElement.innerHTML = `
                        <h2 class="text-lg font-medium text-gray-900 dark:text-white">${data.group.title}</h2>
                        <p class="text-sm text-gray-500 dark:text-gray-400">共 ${data.conversations.length} 轮对话</p>
                        <div class="mt-2 flex justify-center space-x-2">
                            <button id="rename-group" class="px-3 py-1 text-xs text-primary-600 border border-primary-600 rounded-md hover:bg-primary-50 dark:hover:bg-primary-900">
                                重命名对话组
                            </button>
                            <button id="delete-group" class="px-3 py-1 text-xs text-red-600 border border-red-600 rounded-md hover:bg-red-50 dark:hover:bg-red-900">
                                删除对话组
                            </button>
                        </div>
                    `;
                    messageContainer.appendChild(titleElement);
                    
                    // 添加重命名对话组的事件
                    const renameGroupButton = document.getElementById('rename-group');
                    if (renameGroupButton) {
                        renameGroupButton.onclick = function() {
                            const newTitle = prompt('请输入新的对话组标题:', data.group.title);
                            if (newTitle && newTitle.trim()) {
                                updateConversationGroup(groupId, newTitle.trim());
                            }
                        };
                    }
                    
                    // 添加删除对话组的事件
                    const deleteGroupButton = document.getElementById('delete-group');
                    if (deleteGroupButton) {
                        deleteGroupButton.onclick = function() {
                            if (confirm('确定要删除这个对话组吗？此操作不可撤销。')) {
                                deleteConversationGroup(groupId);
                            }
                        };
                    }
                    
                    // 显示所有对话
                    data.conversations.forEach(conv => {
                        // 添加用户消息
                        const userMessageElement = document.createElement('div');
                        userMessageElement.className = 'mb-4';
                        userMessageElement.innerHTML = `
                            <div class="flex items-start justify-end">
                                <div class="mr-3">
                                    <div class="text-sm bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 rounded-lg py-2 px-4 max-w-md">
                                        ${conv.user_message}
                                    </div>
                                    <div class="mt-1 text-xs text-right text-gray-500 dark:text-gray-400">
                                        ${conv.formatted_time}
                                    </div>
                                </div>
                                <div class="flex-shrink-0">
                                    <div class="h-8 w-8 rounded-full bg-primary-500 flex items-center justify-center">
                                        <i class="fas fa-user text-white"></i>
                                    </div>
                                </div>
                            </div>
                        `;
                        messageContainer.appendChild(userMessageElement);
                        
                        // 添加AI响应
                        const aiMessageElement = document.createElement('div');
                        aiMessageElement.className = 'mb-4';
                        aiMessageElement.innerHTML = `
                            <div class="flex items-start">
                                <div class="flex-shrink-0">
                                    <div class="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                                        <i class="fas fa-robot text-primary-600"></i>
                                    </div>
                                </div>
                                <div class="ml-3">
                                    <div class="text-sm text-gray-700 dark:text-gray-300 prose dark:prose-dark max-w-none">
                                        ${md.render(conv.ai_message)}
                                    </div>
                                    <div class="mt-1 text-xs text-gray-500">
                                        tokens: ${conv.tokens} | 费用: ¥${conv.cost.toFixed(6)}
                                    </div>
                                </div>
                            </div>
                        `;
                        messageContainer.appendChild(aiMessageElement);
                    });
                    
                    // 添加"继续对话"提示
                    const continueElement = document.createElement('div');
                    continueElement.className = 'text-center py-4 mt-4 border-t border-gray-200 dark:border-gray-700';
                    continueElement.innerHTML = `
                        <p class="text-sm text-gray-600 dark:text-gray-400">在下方输入框中输入消息，继续这个对话</p>
                    `;
                    messageContainer.appendChild(continueElement);
                    
                    // 高亮代码块
                    messageContainer.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightBlock(block);
                    });
                    
                    // 滚动到底部
                    messageContainer.scrollTop = messageContainer.scrollHeight;
                    
                    // 聚焦输入框
                    if (messageInput) {
                        messageInput.focus();
                    }
                }
            } else {
                // 没有找到对话
                if (messageContainer) {
                    messageContainer.innerHTML = `
                        <div class="flex items-center justify-center h-full">
                            <div class="text-center">
                                <i class="fas fa-exclamation-circle text-3xl text-gray-400 mb-2"></i>
                                <p class="text-gray-600 dark:text-gray-400">该对话组中没有对话</p>
                                <button id="start-group-chat" class="mt-4 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700">
                                    开始对话
                                </button>
                            </div>
                        </div>
                    `;
                    
                    // 添加开始对话的事件
                    const startGroupChatButton = document.getElementById('start-group-chat');
                    if (startGroupChatButton) {
                        startGroupChatButton.onclick = function() {
                            // 清空消息容器，准备开始新对话
                            messageContainer.innerHTML = '';
                            // 添加欢迎消息
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
                            
                            // 聚焦输入框
                            if (messageInput) {
                                messageInput.focus();
                            }
                        };
                    }
                }
                
                if (data.error) {
                    console.error('获取对话组内容错误:', data.error);
                    addConsoleOutput(`[错误] 获取对话组内容错误: ${data.error}`);
                }
            }
        })
        .catch(error => {
            console.error('获取对话组内容失败:', error);
            addConsoleOutput(`[错误] 获取对话组内容失败: ${error.message}`);
            
            // 显示错误信息
            if (messageContainer) {
                messageContainer.innerHTML = `
                    <div class="flex items-center justify-center h-full">
                        <div class="text-center">
                            <i class="fas fa-exclamation-triangle text-3xl text-red-500 mb-2"></i>
                            <p class="text-red-600">加载对话组内容失败: ${error.message}</p>
                        </div>
                    </div>
                `;
            }
        });
}

// 更新对话组
function updateConversationGroup(groupId, title) {
    fetch(`/api/conversation_groups/${groupId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            title: title
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('更新对话组失败: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // 更新页面上的对话组标题
            const titleElement = messageContainer.querySelector('h2');
            if (titleElement) {
                titleElement.textContent = title;
            }
            
            // 更新侧边栏中的对话组标题
            fetchChatHistory();
            
            showToast('对话组已重命名');
        } else {
            throw new Error(data.error || '更新对话组失败');
        }
    })
    .catch(error => {
        console.error('更新对话组失败:', error);
        addConsoleOutput(`[错误] 更新对话组失败: ${error.message}`);
        showToast('更新对话组失败: ' + error.message, 'error');
    });
}

// 删除对话组
function deleteConversationGroup(groupId) {
    fetch(`/api/conversation_groups/${groupId}`, {
        method: 'DELETE'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('删除对话组失败: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // 清空当前对话组ID
            currentGroupId = null;
            
            // 清空消息容器，显示欢迎消息
            if (messageContainer) {
                messageContainer.innerHTML = '';
                // 添加欢迎消息
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
            
            // 更新侧边栏
            fetchChatHistory();
            
            showToast('对话组已删除');
        } else {
            throw new Error(data.error || '删除对话组失败');
        }
    })
    .catch(error => {
        console.error('删除对话组失败:', error);
        addConsoleOutput(`[错误] 删除对话组失败: ${error.message}`);
        showToast('删除对话组失败: ' + error.message, 'error');
    });
}

// 更新统计信息
function updateStats(statsData) {
    if (!statsData) return;
    
    // 更新总费用
    if (statsData.total_cost !== undefined) {
        totalCost = statsData.total_cost;
        const formattedCost = '¥' + totalCost.toFixed(4);
        if (totalCostElement) {
            totalCostElement.textContent = formattedCost;
        }
        if (mobileTotalCostElement) {
            mobileTotalCostElement.textContent = formattedCost;
        }
    }
    
    // 更新对话数量
    fetchStats();
}

// 初始化
document.addEventListener('DOMContentLoaded', init); 