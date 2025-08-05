/**
 * Blyan 공통 헤더 컴포넌트
 */

function createAIBlockHeader(currentPage = '') {
    const pages = {
        'home': { icon: '🏠', name: 'Home', url: 'home.html' },
        'chat': { icon: '💬', name: 'Chat', url: 'chat.html' },
        'contribute': { icon: '⚡', name: 'Join Network', url: 'contribute.html' },
        'explorer': { icon: '🔧', name: 'Technical', url: 'explorer.html' }
    };

    return `
        <header class="Blyan-header">
            <div class="Blyan-nav">
                <div class="Blyan-title">
                    🤖 Blyan
                </div>
                <nav class="nav-tabs">
                    ${Object.entries(pages).map(([key, page]) => `
                        <a href="${page.url}" class="${currentPage === key ? 'active' : ''}">
                            ${page.icon} ${page.name}
                        </a>
                    `).join('')}
                </nav>
                <div class="status-indicators">
                    <div id="api-status" class="status-badge status-offline">
                        API: Offline
                    </div>
                    <div id="pol-status" class="status-badge status-offline">
                        PoL: Unknown
                    </div>
                </div>
            </div>
        </header>
    `;
}

function createPageTabs(tabs, activeTab = '') {
    const tabsHtml = `
        <div class="page-tabs" style="display: flex !important; flex-direction: row !important; width: 100%;">
            ${tabs.map(tab => `
                <button class="page-tab ${activeTab === tab.id ? 'active' : ''}" 
                        onclick="switchTab('${tab.id}')">
                    ${tab.icon} ${tab.name}
                </button>
            `).join('')}
        </div>
    `;
    console.log('Created tabs HTML:', tabsHtml);
    return tabsHtml;
}

// 탭 전환 함수
function switchTab(tabId) {
    console.log('Switching to tab:', tabId);
    
    // 모든 탭 버튼에서 active 제거
    document.querySelectorAll('.page-tab').forEach(tab => {
        tab.classList.remove('active');
        console.log('Removed active from:', tab.textContent);
    });
    
    // 모든 탭 컨텐츠 숨기기
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
        console.log('Hidden content:', content.id);
    });
    
    // 선택된 탭 버튼 활성화
    const selectedTab = document.querySelector(`[onclick="switchTab('${tabId}')"]`);
    if (selectedTab) {
        selectedTab.classList.add('active');
        console.log('Activated tab button:', selectedTab.textContent);
    }
    
    // 선택된 탭 컨텐츠 표시
    const selectedContent = document.getElementById(tabId);
    if (selectedContent) {
        selectedContent.classList.add('active');
        selectedContent.style.display = 'block';
        console.log('Shown content:', tabId);
    }
    
    // DAG 뷰 초기화 (해당 탭인 경우)
    if (tabId === 'dag-view' && window.explorer) {
        window.explorer.initializeDAGView();
    }
}

// API 상태 체크 함수
async function checkAPIStatus() {
    try {
        const response = await fetch(API_CONFIG.baseURL + API_CONFIG.polStatus);
        if (response.ok) {
            const status = await response.json();
            updateStatusBadge('api-status', true, 'API: Online');
            updateStatusBadge('pol-status', status.pol_enabled, 
                `PoL: ${status.pol_enabled ? 'Enabled' : 'Disabled'}`);
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        updateStatusBadge('api-status', false, 'API: Offline');
        updateStatusBadge('pol-status', false, 'PoL: Unknown');
    }
}

function updateStatusBadge(elementId, isOnline, text) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = text;
        element.className = `status-badge ${isOnline ? 'status-online' : 'status-offline'}`;
    }
}

// 페이지 로드시 API 상태 체크
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    setInterval(checkAPIStatus, 30000); // 30초마다 체크
});