/**
 * Language management for Blyan
 */

// Language translations
const translations = {
    en: {
        // Header Navigation
        home: 'Home',
        genesis: 'Genesis',
        howItWorks: 'How it Works',
        chat: 'Chat',
        network: 'Network',
        docs: 'Docs',
        connectWallet: 'Connect Wallet',
        apiStatus: 'API',
        polStatus: 'PoL',
        offline: 'Offline',
        online: 'Online',
        enabled: 'Enabled',
        disabled: 'Disabled',
        unknown: 'Unknown',
        
        // Home Page
        heroMainTitle: 'Decentralized AI for Everyone',
        heroMainSubtitle: 'The world\'s first truly open AI ecosystem',
        heroMainDescription: 'Today, artificial intelligence is advancing rapidly, but the process is concentrated in the hands of a few big tech companies.\nBlyan is building a network where anyone can contribute to and benefit from AI\'s evolution.',
        startChat: 'Start Chat',
        joinNow: 'Join Now',
        
        coreInnovationTitle: 'Our Core Innovation',
        coreInnovationSubtitle: 'Three breakthrough technologies that make decentralized AI possible',
        proofOfLearningTitle: 'Proof-of-Learning',
        proofOfLearningDesc: 'Nodes perform AI model computations instead of hash calculations. When meaningful learning results are accepted, blocks are added and AI grows.',
        distributedGpuTitle: 'Distributed GPU Network',
        distributedGpuDesc: 'A global mesh of independent GPU nodes, not a giant data center, enables AI training and inference.',
        mixtureOfExpertsTitle: 'Mixture-of-Experts + DAG',
        mixtureOfExpertsDesc: 'Dynamic architecture routes tasks to optimal expert models, creating efficient AI that everyone can participate in.',
        
        // Home Page - Additional Sections
        whyItMattersTitle: 'Why Blyan?',
        whyItMattersSubtitle: 'The world\'s first truly open AI ecosystem',
        transparencyTitle: 'Transparency',
        transparencyDesc: 'All model updates are verifiable on-chain. Let\'s check the AI evolution process together.',
        accessibilityTitle: 'Accessibility', 
        accessibilityDesc: 'Anyone can participate—from professional researchers to curious individuals.',
        collectiveOwnershipTitle: 'Collective Ownership',
        collectiveOwnershipDesc: 'Together, AI can belong to all of humanity.',
        joinMovementTitle: 'Join Blyan',
        joinMovementDesc: 'Blyan is not just another blockchain or AI startup.\nIt\'s about creating artificial intelligence of everyone, for everyone, with everyone.',
        bePartOfFuture: 'Let\'s build the future of AI together',
        
        // Chat Page
        aiChat: 'Chat with Blyan',
        chatWithDistributed: 'Chat with decentralized AI network',
        aiAssistant: 'Blyan AI',
        aiWelcomeMessage: "Hello! I'm powered by the Blyan decentralized network. Ask me anything!",
        aiThinking: 'Blyan is thinking...',
        typeYourMessage: 'Type your message...',
        send: 'Send',
        sendMessage: 'Send Message',
        chats: 'Chat History',
        newChat: 'New Chat',
        welcomeToBlyanchat: 'Welcome to Blyan AI',
        welcomeChatDesc: 'Start a conversation with our decentralized AI assistant. Your questions will be processed by our global network of expert nodes.',
        freeChatsRemaining: '5 free chats remaining',
        
        // Chat errors and messages
        connectionError: 'Connection error. Please check your network and try again.',
        errorEncountered: 'Sorry, I encountered an error. Please try again.',
        requestFailed: 'Request failed',
        serverError: 'Server error. Please try again later.',
        refreshToRetry: 'Please refresh the page to continue.',
        sessionExpired: 'Session Expired',
        refreshPage: 'Refresh Page',
        freeTierLimitReached: 'You have reached the free tier limit of 5 messages. Please connect a wallet to continue.',
        authenticationRequired: 'Authentication required. Please try again later.',
        newConversation: 'New Conversation',
        continueConversation: 'Continue Conversation',
        startFreshConversation: 'Start a fresh conversation with Blyan AI.',
        continueYourConversation: 'Continue your conversation with Blyan AI.',
        noPreviousConversations: 'No previous conversations',
        
        // Contribute Page
        systemRequirements: 'System Requirements',
        systemRequirementsDesc: 'What you need to join the network',
        gpu: 'GPU',
        vram: 'VRAM', 
        internet: 'Internet',
        software: 'Software',
        quickSetupGuide: 'Quick Setup Guide',
        quickSetupDesc: 'Join the network in 3 simple steps',
        installPrerequisites: 'Install Prerequisites',
        installPrereqDesc: 'Ensure Docker and NVIDIA Container Toolkit are installed on your system.',
        generateNodeKey: 'Generate Your Node Key',
        generateNodeKeyDesc: 'Click the button below to generate a unique API key for your node. This key identifies your contributions.',
        generateNodeKeyBtn: 'Generate Node Key',
        runYourNode: 'Run Your Node',
        runYourNodeDesc: 'Copy and run the Docker command with your generated key to start contributing.',
        earningPotential: 'Earning Potential',
        earningPotentialDesc: 'Estimated rewards based on GPU type',
        
        // Join Modal
        joinTheNetwork: 'Join the Network',
        joinModalDesc: 'We will create a node operator API key and show a one‑line command to start a GPU node. CPU mode is not supported for LLMs.',
        generateNodeKeyModal: 'Generate Node Key',
        provisioning: 'Provisioning…',
        failedToProvisionKey: 'Failed to provision key',
        copyRunCommand: 'Copy and run the command on a GPU machine (Docker required).',
        orSetEnvAndRun: 'Or set the key as env and run the Python node.'
    },
    
    ko: {
        // Header Navigation
        home: '홈',
        howItWorks: '작동 원리',
        chat: '채팅',
        network: '네트워크',
        docs: '문서',
        connectWallet: '지갑 연결',
        apiStatus: 'API',
        polStatus: 'PoL',
        offline: '오프라인',
        online: '온라인',
        enabled: '활성화',
        disabled: '비활성화',
        unknown: '알 수 없음',
        
        // Home Page
        heroMainTitle: '모두를 위한 탈중앙화 AI',
        heroMainSubtitle: '세계 최초 진정한 오픈 AI 생태계',
        heroMainDescription: '오늘날 인공지능은 눈부시게 발전하고 있지만, 그 과정은 소수의 빅테크 기업에 집중되어 있습니다.\nBlyan은 누구나 AI의 진화에 기여하고 혜택을 받을 수 있는 네트워크를 구축하고 있습니다.',
        startChat: '채팅 시작',
        joinNow: '지금 참여',
        
        coreInnovationTitle: '핵심 혁신 기술',
        coreInnovationSubtitle: '탈중앙화 AI를 가능하게 하는 세 가지 혁신 기술',
        proofOfLearningTitle: 'Proof-of-Learning',
        proofOfLearningDesc: '노드들은 해시 계산 대신 AI 모델의 계산을 수행합니다. 의미있는 학습 결과가 수용되면 블록이 추가되며 AI가 성장하게 됩니다',
        distributedGpuTitle: '분산 GPU 네트워크',
        distributedGpuDesc: '거대한 데이터센터가 아닌 독립적인 GPU 노드들의 글로벌 메시가 AI 훈련 및 추론을 가능하게 합니다.',
        mixtureOfExpertsTitle: 'Mixture-of-Experts + DAG',
        mixtureOfExpertsDesc: '동적 아키텍처가 최적의 전문가 모델로 작업을 라우팅하여, 모두가 참여할 수 있는 효율적인 AI를 구성합니다.',
        
        // Home Page - Additional Sections
        whyItMattersTitle: '왜 Blyan인가',
        whyItMattersSubtitle: '모두의, 모두를 위한, 모두가 함께 하는 AI',
        transparencyTitle: '투명성',
        transparencyDesc: '모든 모델 업데이트는 체인에서 검증 가능합니다. AI의 진화 과정을 같이 확인해보세요.',
        accessibilityTitle: '접근성', 
        accessibilityDesc: '전문 연구자부터 호기심 많은 일반인까지 누구나 참여할 수 있습니다.',
        collectiveOwnershipTitle: '집단 소유',
        collectiveOwnershipDesc: '함께 하면, AI는 모두를 위한 것이 될 수 있습니다.',
        joinMovementTitle: 'Blyan에 동참하세요',
        joinMovementDesc: 'Blyan은 단순한 블록체인이나 AI 스타트업이 아닙니다.\n모두의, 모두를 위한, 모두가 함께 하는, 인공지능을 만드는 일입니다.',
        bePartOfFuture: '미래의 AI를 함께 만들어보세요',
        
        // Chat Page
        aiChat: 'Blyan과 채팅',
        chatWithDistributed: 'Blyan 분산 네트워크와 대화하기',
        aiAssistant: 'Blyan',
        aiWelcomeMessage: '안녕하세요! 저는 Blyan 분산 네트워크로 구동됩니다. 무엇이든 물어보세요!',
        aiThinking: 'Blyan이 생각 중입니다...',
        typeYourMessage: '메시지를 입력하세요...',
        send: '전송',
        sendMessage: '메시지 전송',
        chats: '채팅 기록',
        newChat: '새로운 채팅',
        welcomeToBlyanchat: 'Blyan AI에 오신 것을 환영합니다',
        welcomeChatDesc: '탈중앙화된 AI 어시스턴트와 대화를 시작하세요. 질문은 전 세계 전문가 노드 네트워크에서 처리됩니다.',
        freeChatsRemaining: '5개의 무료 채팅이 남았습니다',
        
        // Chat errors and messages
        connectionError: '연결 오류가 발생했습니다. 네트워크를 확인하고 다시 시도해주세요.',
        errorEncountered: '죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.',
        requestFailed: '요청 실패',
        serverError: '서버 오류입니다. 나중에 다시 시도해주세요.',
        refreshToRetry: '계속하려면 페이지를 새로고침해주세요.',
        sessionExpired: '세션 만료',
        refreshPage: '페이지 새로고침',
        freeTierLimitReached: '무료 이용 한도인 5개 메시지에 도달했습니다. 계속하려면 지갑을 연결해주세요.',
        authenticationRequired: '인증이 필요합니다. 나중에 다시 시도해주세요.',
        newConversation: '새로운 대화',
        continueConversation: '대화 계속하기',
        startFreshConversation: 'Blyan AI와 새로운 대화를 시작하세요.',
        continueYourConversation: 'Blyan AI와 대화를 계속하세요.',
        noPreviousConversations: '이전 대화가 없습니다',
        
        // Contribute Page
        systemRequirements: '시스템 요구사항',
        systemRequirementsDesc: '네트워크 참여에 필요한 사양',
        gpu: 'GPU',
        vram: 'VRAM', 
        internet: '인터넷',
        software: '소프트웨어',
        quickSetupGuide: '빠른 설정 가이드',
        quickSetupDesc: '3단계로 네트워크에 참여하기',
        installPrerequisites: '필수 요소 설치',
        installPrereqDesc: 'Docker와 NVIDIA Container Toolkit이 시스템에 설치되어 있는지 확인하세요.',
        generateNodeKey: '노드 키 생성',
        generateNodeKeyDesc: '아래 버튼을 클릭하여 노드용 고유 API 키를 생성하세요. 이 키는 귀하의 기여를 식별합니다.',
        generateNodeKeyBtn: '노드 키 생성',
        runYourNode: '노드 실행',
        runYourNodeDesc: '생성된 키로 Docker 명령어를 복사하고 실행하여 기여를 시작하세요.',
        earningPotential: '수익 가능성',
        earningPotentialDesc: 'GPU 유형별 예상 보상',
        
        // Join Modal
        joinTheNetwork: '네트워크 참여',
        joinModalDesc: '노드 운영자 API 키를 생성하고 GPU 노드를 시작하는 한 줄 명령어를 제공합니다. LLM의 경우 CPU 모드는 지원되지 않습니다.',
        generateNodeKeyModal: '노드 키 생성',
        provisioning: '생성 중…',
        failedToProvisionKey: '키 생성 실패',
        copyRunCommand: 'GPU 머신에서 명령어를 복사하고 실행하세요 (Docker 필요).',
        orSetEnvAndRun: '또는 키를 환경변수로 설정하고 Python 노드를 실행하세요.'
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