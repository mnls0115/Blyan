/**
 * Language management for Blyan
 */

// Language translations
const translations = {
    en: {
        // Header
        home: 'Home',
        chat: 'Chat',
        joinNetwork: 'Join Network',
        technical: 'Technical',
        connectWallet: 'Connect Wallet',
        apiStatus: 'API',
        polStatus: 'PoL',
        offline: 'Offline',
        online: 'Online',
        enabled: 'Enabled',
        disabled: 'Disabled',
        unknown: 'Unknown',
        
        // Home page - Hero section
        heroTitle: 'Blyan',
        heroSubtitle: "The World's First Decentralized AI Network",
        heroDescription: 'Experience AI powered by collective intelligence. Chat with AI or contribute your computing power to earn rewards.',
        
        // Vision section (new)
        visionTitle: 'Why We Built Blyan',
        visionTrustworthy: 'Creating Trustworthy AI',
        visionTrustworthyDesc: 'In a world of black-box AI systems, Blyan offers transparency through blockchain-verified computation.',
        visionDemocratic: 'Democratizing AI Development',
        visionDemocraticDesc: 'Not just for tech giants anymore. Together, we can build and evolve AI models collectively.',
        visionEvolution: 'Blockchain Evolution',
        visionEvolutionDesc: 'Beyond transaction records - a living, learning infrastructure for distributed intelligence.',
        visionComparison: 'Did you know? The energy Bitcoin uses in just 3 hours could train a GPT-4 level AI model. With Proof-of-Learning, we turn computation into intelligence.',
        
        // Action cards
        chatWithAI: 'Chat with AI',
        chatDescription: 'Start a conversation with our decentralized AI. Powered by distributed experts across the network.',
        startChatting: 'Start Chatting',
        uploadDataset: 'Upload Dataset',
        uploadDescription: 'Contribute training data to improve AI models. Zero-cost participation with democratic governance.',
        uploadNow: 'Upload Now',
        joinTheNetwork: 'Join the Network',
        joinDescription: 'Share your computing power and earn rewards. One-click setup to become part of the AI network.',
        joinNow: 'Join Now',
        datasetExplorer: 'Dataset Explorer',
        explorerDescription: 'Explore community datasets, vote on proposals, and track data governance in real-time.',
        exploreDatasets: 'Explore Datasets',
        
        // Network status
        networkStatus: 'Network Status',
        activeExperts: 'Active Experts',
        networkNodes: 'Network Nodes',
        totalRequests: 'Total Requests',
        networkHealth: 'Network Health',
        
        // Footer
        advancedFeatures: 'Looking for advanced features?',
        viewDashboard: 'View Technical Dashboard & Analytics →',
        
        // Chat page
        aiChat: 'AI Chat',
        chatWithDistributed: 'Chat with our decentralized AI network',
        aiAssistant: 'AI Assistant',
        aiWelcomeMessage: "Hello! I'm powered by the Blyan decentralized network. Ask me anything!",
        aiThinking: 'AI is thinking...',
        typeYourMessage: 'Type your message...',
        send: 'Send',
        sendMessage: 'Send Message',
        
        // Join Network page
        joinNetworkTitle: 'Join the Blyan Network',
        shareComputingPower: 'Share your computing power and earn rewards',
        systemCheck: 'System Check',
        checkRequirements: "Let's check if your computer meets the requirements:",
        cpuRequirement: 'CPU: Multi-core processor',
        memoryRequirement: 'Memory: At least 4GB RAM available',
        networkRequirement: 'Network: Stable internet connection',
        gpuRequirement: 'GPU: Checking for AI acceleration hardware',
        browserRequirement: 'Browser: Modern browser with WebAssembly support',
        checkMySystem: 'Check My System',
        startYourNode: 'Start Your Node',
        oneClickSetup: 'One-click setup to join the network as an expert node:',
        nodeName: 'Node Name:',
        processingMode: 'Processing Mode:',
        cpuLight: 'CPU Only - 25% (Light)',
        cpuBalanced: 'CPU Only - 50% (Balanced)',
        cpuHigh: 'CPU Only - 75% (High)',
        gpuLight: 'GPU Accelerated - 25%',
        gpuBalanced: 'GPU Accelerated - 50%',
        gpuHigh: 'GPU Accelerated - 75%',
        startMyNode: '🚀 Start My Node',
        nodeRunning: '✅ Node Running Successfully!',
        nodeId: 'Node ID:',
        status: 'Status:',
        active: 'Active',
        requestsServed: 'Requests Served:',
        disconnectNode: '🔌 Disconnect Node',
        earnRewards: 'Earn Rewards',
        estimatedEarnings: 'Your estimated earnings based on network activity:',
        perDay: 'per day (estimated)',
        earningsNote: '💡 Earnings based on RTX 3090 benchmarks (~$210/day for 10 GPUs). Your actual earnings may vary.',
        advancedOptions: 'Advanced Options',
        wantMoreControl: 'Want more control? Check out the',
        technicalDashboard: 'technical dashboard',
        expertNodeManagement: 'for expert node management and detailed analytics.',
        
        // Technical/Explorer page
        explorerTitle: 'Blyan Explorer',
        blockchainOverview: 'Blockchain Overview',
        expertManagement: 'Expert Management',
        expertAnalytics: 'Expert Analytics',
        dagView: 'DAG View',
        p2pNetwork: 'P2P Network',
        
        // Blockchain Overview tab
        metaChain: 'Meta-chain (Architecture)',
        parameterChain: 'Parameter-chain (Experts)',
        chainHeight: 'Chain Height',
        totalBlocks: 'Total Blocks',
        modelArchitecture: 'Model Architecture',
        modelName: 'Model Name',
        layers: 'Layers',
        expertsPerLayer: 'Experts per Layer',
        routingStrategy: 'Routing Strategy',
        hiddenSize: 'Hidden Size',
        metaBlockDetails: 'Meta Block Details',
        blockHash: 'Block Hash',
        index: 'Index',
        
        // Parameter Chain tab
        expertBlocks: 'Expert Blocks',
        searchExpert: 'Search Expert',
        layer: 'Layer',
        blockType: 'Block Type',
        allLayers: 'All Layers',
        allTypes: 'All Types',
        expert: 'Expert',
        router: 'Router',
        allStatus: 'All Status',
        filter: '🔍 Filter',
        clear: 'Clear',
        expertName: 'Expert Name',
        type: 'Type',
        size: 'Size (KB)',
        usageCount: 'Usage Count',
        deltaScore: 'Δ Score',
        actions: 'Actions',
        loadingExpertBlocks: 'Loading expert blocks...',
        
        // Expert Analytics tab
        expertPerformanceAnalytics: 'Expert Performance Analytics',
        topPerformingExperts: 'Top Performing Experts',
        expertUsageDistribution: 'Expert Usage Distribution',
        averageLatency: 'Average Latency',
        totalInferenceRequests: 'Total Inference Requests',
        activeExperts: 'Active Experts',
        
        // P2P Network tab
        registeredNodes: 'Registered Nodes',
        nodeManagement: 'Node Management',
        expertDistribution: 'Expert Distribution',
        networkTopology: 'Network Topology',
        
        // Common
        loading: 'Loading...',
        viewDetails: 'View Details',
        download: 'Download',
        noDataAvailable: 'No data available'
    },
    
    ko: {
        // Header
        home: '홈',
        chat: '채팅',
        joinNetwork: '네트워크 참여',
        technical: '기술 정보',
        connectWallet: '지갑 연결',
        apiStatus: 'API',
        polStatus: 'PoL',
        offline: '오프라인',
        online: '온라인',
        enabled: '활성화',
        disabled: '비활성화',
        unknown: '알 수 없음',
        
        // Home page - Hero section
        heroTitle: 'Blyan',
        heroSubtitle: '세계 최초 분산형 AI 네트워크',
        heroDescription: '집단 지성으로 구동되는 AI를 경험하세요. AI와 대화하거나 컴퓨팅 파워를 제공하여 보상을 받으세요.',
        
        // Vision section (new)
        visionTitle: '왜 Blyan을 만들었나',
        visionTrustworthy: '신뢰할 수 있는 AI',
        visionTrustworthyDesc: '블랙박스 AI 시스템의 시대에, Blyan은 블록체인 검증을 통한 투명성을 제공합니다.',
        visionDemocratic: 'AI 개발의 민주화',
        visionDemocraticDesc: '더 이상 대기업만의 전유물이 아닙니다. 함께라면 우리도 AI 모델을 만들고 발전시킬 수 있습니다.',
        visionEvolution: '블록체인의 진화',
        visionEvolutionDesc: '단순한 거래 기록을 넘어 - 분산 지능을 위한 살아있는 학습 인프라입니다.',
        visionComparison: '알고 계셨나요? 비트코인이 단 3시간 동안 사용하는 에너지로 GPT-4 수준의 AI 모델을 학습시킬 수 있습니다. Proof-of-Learning으로 우리는 연산을 지능으로 바꿉니다.',
        
        // Action cards
        chatWithAI: 'AI와 대화하기',
        chatDescription: '분산형 AI와 대화를 시작하세요. 네트워크 전반의 분산 전문가들이 제공합니다.',
        startChatting: '채팅 시작',
        uploadDataset: '데이터셋 업로드',
        uploadDescription: 'AI 모델 개선을 위한 학습 데이터를 기여하세요. 민주적 거버넌스와 함께 무료로 참여 가능합니다.',
        uploadNow: '지금 업로드',
        joinTheNetwork: '네트워크 참여',
        joinDescription: '컴퓨팅 파워를 공유하고 보상을 받으세요. 원클릭으로 AI 네트워크의 일원이 되세요.',
        joinNow: '지금 참여',
        datasetExplorer: '데이터셋 탐색기',
        explorerDescription: '커뮤니티 데이터셋을 탐색하고, 제안에 투표하며, 실시간으로 데이터 거버넌스를 추적하세요.',
        exploreDatasets: '데이터셋 탐색',
        
        // Network status
        networkStatus: '네트워크 상태',
        activeExperts: '활성 전문가',
        networkNodes: '네트워크 노드',
        totalRequests: '총 요청 수',
        networkHealth: '네트워크 상태',
        
        // Footer
        advancedFeatures: '고급 기능을 찾고 계신가요?',
        viewDashboard: '기술 대시보드 및 분석 보기 →',
        
        // Chat page
        aiChat: 'AI 채팅',
        chatWithDistributed: '분산형 AI 네트워크와 대화하기',
        aiAssistant: 'AI 어시스턴트',
        aiWelcomeMessage: '안녕하세요! 저는 Blyan 분산 네트워크로 구동됩니다. 무엇이든 물어보세요!',
        aiThinking: 'AI가 생각 중입니다...',
        typeYourMessage: '메시지를 입력하세요...',
        send: '전송',
        sendMessage: '메시지 전송',
        
        // Join Network page
        joinNetworkTitle: 'Blyan 네트워크 참여',
        shareComputingPower: '컴퓨팅 파워를 공유하고 보상을 받으세요',
        systemCheck: '시스템 확인',
        checkRequirements: '컴퓨터가 요구 사항을 충족하는지 확인해보겠습니다:',
        cpuRequirement: 'CPU: 멀티코어 프로세서',
        memoryRequirement: '메모리: 최소 4GB RAM 필요',
        networkRequirement: '네트워크: 안정적인 인터넷 연결',
        gpuRequirement: 'GPU: AI 가속 하드웨어 확인 중',
        browserRequirement: '브라우저: WebAssembly 지원 최신 브라우저',
        checkMySystem: '시스템 확인하기',
        startYourNode: '노드 시작하기',
        oneClickSetup: '원클릭으로 전문가 노드로 네트워크에 참여:',
        nodeName: '노드 이름:',
        processingMode: '처리 모드:',
        cpuLight: 'CPU 전용 - 25% (가벼움)',
        cpuBalanced: 'CPU 전용 - 50% (균형)',
        cpuHigh: 'CPU 전용 - 75% (높음)',
        gpuLight: 'GPU 가속 - 25%',
        gpuBalanced: 'GPU 가속 - 50%',
        gpuHigh: 'GPU 가속 - 75%',
        startMyNode: '🚀 내 노드 시작',
        nodeRunning: '✅ 노드가 성공적으로 실행 중입니다!',
        nodeId: '노드 ID:',
        status: '상태:',
        active: '활성',
        requestsServed: '처리된 요청:',
        disconnectNode: '🔌 노드 연결 해제',
        earnRewards: '보상 획득',
        estimatedEarnings: '네트워크 활동 기반 예상 수익:',
        perDay: '일일 (예상)',
        earningsNote: '💡 RTX 3090 벤치마크 기준 수익 (10개 GPU로 ~$210/일). 실제 수익은 다를 수 있습니다.',
        advancedOptions: '고급 옵션',
        wantMoreControl: '더 많은 제어가 필요하신가요?',
        technicalDashboard: '기술 대시보드',
        expertNodeManagement: '에서 전문가 노드 관리 및 상세 분석을 확인하세요.',
        
        // Technical/Explorer page
        explorerTitle: 'Blyan 탐색기',
        blockchainOverview: '블록체인 개요',
        expertManagement: '전문가 관리',
        expertAnalytics: '전문가 분석',
        dagView: 'DAG 뷰',
        p2pNetwork: 'P2P 네트워크',
        
        // Blockchain Overview tab
        metaChain: '메타 체인 (아키텍처)',
        parameterChain: '파라미터 체인 (전문가)',
        chainHeight: '체인 높이',
        totalBlocks: '전체 블록',
        modelArchitecture: '모델 아키텍처',
        modelName: '모델 이름',
        layers: '레이어',
        expertsPerLayer: '레이어당 전문가',
        routingStrategy: '라우팅 전략',
        hiddenSize: '히든 크기',
        metaBlockDetails: '메타 블록 상세',
        blockHash: '블록 해시',
        index: '인덱스',
        
        // Parameter Chain tab
        expertBlocks: '전문가 블록',
        searchExpert: '전문가 검색',
        layer: '레이어',
        blockType: '블록 타입',
        allLayers: '모든 레이어',
        allTypes: '모든 타입',
        expert: '전문가',
        router: '라우터',
        allStatus: '모든 상태',
        filter: '🔍 필터',
        clear: '초기화',
        expertName: '전문가 이름',
        type: '타입',
        size: '크기 (KB)',
        usageCount: '사용 횟수',
        deltaScore: 'Δ 점수',
        actions: '작업',
        loadingExpertBlocks: '전문가 블록 로딩 중...',
        
        // Expert Analytics tab
        expertPerformanceAnalytics: '전문가 성능 분석',
        topPerformingExperts: '최고 성능 전문가',
        expertUsageDistribution: '전문가 사용 분포',
        averageLatency: '평균 지연시간',
        totalInferenceRequests: '전체 추론 요청',
        activeExperts: '활성 전문가',
        
        // P2P Network tab
        registeredNodes: '등록된 노드',
        nodeManagement: '노드 관리',
        expertDistribution: '전문가 분포',
        networkTopology: '네트워크 토폴로지',
        
        // Common
        loading: '로딩 중...',
        viewDetails: '상세 보기',
        download: '다운로드',
        noDataAvailable: '데이터 없음'
    }
};

// Current language state
let currentLanguage = localStorage.getItem('blyan-language') || 'en';

// Get translation
function t(key) {
    return translations[currentLanguage][key] || translations['en'][key] || key;
}

// Update all translatable elements
function updatePageLanguage() {
    // Update elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        element.textContent = t(key);
    });
    
    // Update elements with data-i18n-placeholder attribute
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        element.placeholder = t(key);
    });
    
    // Update page title if it has a translation key
    const titleElement = document.querySelector('title');
    if (titleElement && titleElement.hasAttribute('data-i18n')) {
        const key = titleElement.getAttribute('data-i18n');
        titleElement.textContent = t(key);
    }
}

// Change language
function changeLanguage(lang) {
    if (translations[lang]) {
        currentLanguage = lang;
        localStorage.setItem('blyan-language', lang);
        updatePageLanguage();
        
        // Refresh header if it exists
        if (typeof refreshHeader === 'function') {
            refreshHeader();
        }
    }
}

// Initialize language on page load
document.addEventListener('DOMContentLoaded', () => {
    updatePageLanguage();
});

// Create language selector HTML
function createLanguageSelector() {
    return `
        <select id="language-selector" class="language-selector" onchange="changeLanguage(this.value)">
            <option value="en" ${currentLanguage === 'en' ? 'selected' : ''}>🇺🇸 English</option>
            <option value="ko" ${currentLanguage === 'ko' ? 'selected' : ''}>🇰🇷 한국어</option>
        </select>
    `;
}