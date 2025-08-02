const API_URL = API_CONFIG.baseURL + API_CONFIG.chat;
const BALANCE_URL = API_CONFIG.baseURL + API_CONFIG.balance;
const CHAIN_URL = API_CONFIG.baseURL + API_CONFIG.chain;

let form, input, messages, checkBalanceBtn, walletInput, balanceDisplay, refreshChainBtn, blocksList;

function initializeElements() {
    form = document.getElementById('chat-form');
    input = document.getElementById('prompt-input');
    messages = document.getElementById('message-list');
    checkBalanceBtn = document.getElementById('check-balance');
    walletInput = document.getElementById('wallet-address');
    balanceDisplay = document.getElementById('balance-display');
    refreshChainBtn = document.getElementById('refresh-chain');
    blocksList = document.getElementById('blocks-list');
}

function attachEventListeners() {
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const prompt = input.value.trim();
            if (!prompt) return;

            addMessage('You', prompt, 'user');
            input.value = '';
            input.focus();

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });

                if (!response.ok) throw new Error('Network response was not ok');
                
                const data = await response.json();
                addMessage('AI', data.response || 'No response', 'ai');
            } catch (error) {
                console.error('Error:', error);
                addMessage('AI', 'Sorry, there was an error processing your request.', 'ai error');
            }
        });
    }

    if (checkBalanceBtn) {
        checkBalanceBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            const addr = walletInput.value.trim();
            if (!addr) return;
            try {
                const res = await fetch(BALANCE_URL + encodeURIComponent(addr));
                if (!res.ok) throw new Error('Failed');
                const data = await res.json();
                balanceDisplay.textContent = `Balance: ${data.balance}`;
            } catch (err) {
                balanceDisplay.textContent = 'Error loading balance';
            }
        });
    }

    if (refreshChainBtn) {
        refreshChainBtn.addEventListener('click', (e) => { 
            e.preventDefault(); 
            loadBlocks(); 
        });
        // initial load
        loadBlocks();
    }
}

async function loadBlocks() {
    if (!blocksList) return;
    
    try {
        const res = await fetch(CHAIN_URL);
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        
        blocksList.innerHTML = '';
        
        data.blocks.forEach(block => {
            const li = document.createElement('li');
            li.innerHTML = `
                <strong>Block ${block.index}</strong><br>
                Hash: ${block.hash.substring(0, 16)}...<br>
                Size: ${block.payload_size} bytes
            `;
            li.style.marginBottom = '10px';
            li.style.padding = '10px';
            li.style.border = '1px solid #ddd';
            li.style.borderRadius = '4px';
            blocksList.appendChild(li);
        });
    } catch (error) {
        blocksList.innerHTML = '<li>Error loading blocks</li>';
    }
}

function addMessage(sender, text, type) {
    if (!messages) return;
    
    const li = document.createElement('li');
    li.className = `message ${type}`;
    li.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messages.appendChild(li);
    messages.parentElement.scrollTop = messages.parentElement.scrollHeight;
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 헤더와 탭이 생성된 후 요소 초기화
    setTimeout(() => {
        initializeElements();
        attachEventListeners();
    }, 100);
});