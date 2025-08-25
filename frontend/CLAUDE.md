# Frontend Module Guidelines

## Overview
Web interface for Blyan Network - vanilla JavaScript, no build step required.

## Key Files
- `index.html` - Main landing page
- `chat.html` - Chat interface
- `explorer.html` - Blockchain explorer
- `wallet.js` - MetaMask integration
- `common-header.js` - Shared navigation
- `config.js` - API configuration

## Critical Reminders

### Production Code Standards
- **NO MOCK DATA**: Never display placeholder text or fake responses
- **NO HARDCODED URLS**: Always use config.js for API endpoints
- **REAL ERROR HANDLING**: Show meaningful errors to users
- **NO DEBUG CODE**: Remove all console.log before production

### API Configuration
```javascript
// config.js - Use environment-based configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000' 
  : 'https://blyan.com/api';

// ❌ NEVER hardcode URLs
fetch('http://localhost:8000/chat')  // WRONG!

// ✅ Always use config
fetch(`${API_BASE_URL}/chat`)  // CORRECT
```

### No Mock Responses
```javascript
// ❌ FORBIDDEN - Never show fake data
function displayResponse() {
    chatDiv.innerHTML = "This is a test response";  // NEVER!
}

// ✅ REQUIRED - Always use real API responses
async function displayResponse(prompt) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt})
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        chatDiv.innerHTML = sanitizeHTML(data.text);
    } catch (error) {
        showError('Unable to get response. Please try again.');
    }
}
```

### Security
- NEVER expose API keys in frontend
- Use secure WebSocket (wss://) in production
- Validate all user inputs
- Sanitize HTML to prevent XSS

### MetaMask Integration
```javascript
// Connect wallet
if (window.ethereum) {
    const accounts = await window.ethereum.request({
        method: 'eth_requestAccounts'
    });
}
```

## Common Patterns

### API Calls
```javascript
// With error handling
try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt: userInput})
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
} catch (error) {
    console.error('API Error:', error);
    showError('Service temporarily unavailable');
}
```

### Streaming Responses
```javascript
const eventSource = new EventSource(`${API_BASE_URL}/chat/stream?prompt=${encodeURIComponent(prompt)}`);
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    appendToken(data.token);
};
```

## Testing
- Open HTML files directly in browser
- Use browser DevTools for debugging
- Test with both localhost and production API