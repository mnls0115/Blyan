const form = document.getElementById('chat-form');
const input = document.getElementById('prompt-input');
const messages = document.getElementById('message-list');
const API_URL = 'http://127.0.0.1:8000/chat';

const BALANCE_URL = 'http://127.0.0.1:8000/balance/';
const CHAIN_URL = 'http://127.0.0.1:8000/chain/B/blocks?limit=10';

const checkBalanceBtn = document.getElementById('check-balance');
const walletInput = document.getElementById('wallet-address');
const balanceDisplay = document.getElementById('balance-display');
const refreshChainBtn = document.getElementById('refresh-chain');
const blocksList = document.getElementById('blocks-list');

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
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'API Error');
        }

        const data = await response.json();
        addMessage('AI', data.response, 'ai');
    } catch (error) {
        addMessage('Error', error.message, 'error');
    }
});

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
        balanceDisplay.textContent = 'Error';
    }
});

async function loadBlocks() {
    try {
        const res = await fetch(CHAIN_URL);
        if (!res.ok) throw new Error('failed');
        const data = await res.json();
        blocksList.innerHTML = '';
        data.blocks.forEach((blk) => {
            const li = document.createElement('li');
            li.textContent = `#${blk.index} size=${blk.payload_size}`;
            blocksList.appendChild(li);
        });
    } catch (err) {
        blocksList.innerHTML = '<li>Error fetching blocks</li>';
    }
}

refreshChainBtn.addEventListener('click', (e) => { e.preventDefault(); loadBlocks(); });
// initial load
loadBlocks();

function addMessage(sender, text, type) {
    const li = document.createElement('li');
    li.className = `message ${type}`;
    li.innerHTML = `<strong>${sender}:</strong> ${text}`;
    messages.appendChild(li);
    messages.parentElement.scrollTop = messages.parentElement.scrollHeight;
} 