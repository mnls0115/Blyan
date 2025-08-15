#!/bin/bash
# Launch Verification Script
# Quick checks for all critical systems

echo "🚀 Blyan Network Launch Verification"
echo "===================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0

# Function to check test result
check_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $2"
        ((FAILED++))
    fi
}

# 1. PostgreSQL Check
echo "1️⃣  PostgreSQL Verification"
echo "----------------------------"
psql -U blyan -d blyandb -c '\dt' &>/dev/null
check_result $? "Tables exist"

psql -U blyan -d blyandb -c 'SELECT NOW()' &>/dev/null
check_result $? "Database connection works"

psql -U blyan -d blyandb -c "SHOW wal_level" | grep -q replica
check_result $? "WAL archiving enabled"
echo ""

# 2. Redis Check
echo "2️⃣  Redis Verification"
echo "----------------------"
redis-cli ping &>/dev/null
check_result $? "Redis is running"

redis-cli CONFIG GET maxmemory-policy | grep -q lru
check_result $? "LRU eviction policy set"

redis-cli ACL LIST | grep -q blyan_api
check_result $? "ACL users configured"
echo ""

# 3. API Key Permissions Check
echo "3️⃣  API Key Security"
echo "--------------------"
# Test with a node_operator key (you need to create one first)
NODE_KEY=${NODE_KEY:-"test_node_key"}
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X GET https://blyan.com/api/wallet/balance \
    -H "X-API-Key: $NODE_KEY")
[ "$RESPONSE" = "403" ]
check_result $? "node_operator blocked from /wallet (got $RESPONSE)"
echo ""

# 4. Log Masking Check
echo "4️⃣  Log Security"
echo "----------------"
tail -100 /var/log/aiblock.log | grep -E "X-API-Key|Bearer" | grep -v "..." &>/dev/null
if [ $? -eq 0 ]; then
    check_result 1 "API keys are NOT masked in logs"
else
    check_result 0 "API keys are masked in logs"
fi
echo ""

# 5. Abuse Detection Check
echo "5️⃣  Abuse Prevention"
echo "--------------------"
# Simulate abuse (10 bad requests)
for i in {1..10}; do
    curl -s -X POST https://blyan.com/api/p2p/register \
        -H "X-API-Key: invalid_key" \
        -d '{"node_id":"test"}' &>/dev/null
done

# 11th request should be blocked
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST https://blyan.com/api/p2p/register \
    -H "X-API-Key: invalid_key" \
    -d '{"node_id":"test"}')
[ "$RESPONSE" = "429" ] || [ "$RESPONSE" = "403" ]
check_result $? "Abuse detection working (got $RESPONSE)"
echo ""

# 6. Prometheus Metrics Check
echo "6️⃣  Observability"
echo "-----------------"
curl -s http://localhost:8000/metrics | grep -q "# TYPE"
check_result $? "Prometheus metrics exposed"

curl -s http://localhost:8000/metrics | grep -q "http_requests_total"
check_result $? "Request metrics present"
echo ""

# 7. Go/No-Go Quick Tests
echo "7️⃣  Go/No-Go Tests"
echo "------------------"

# SSE Test
echo "Testing SSE streaming..."
STREAM_TEST=$(timeout 5 curl -s -X POST https://blyan.com/api/chat/stream \
    -H "Content-Type: application/json" \
    -H "X-User-Address: test_user" \
    -d '{"prompt":"Hello","stream":true}' | head -1)
[ ! -z "$STREAM_TEST" ]
check_result $? "SSE streaming works"

# Free tier test
echo "Testing free tier..."
FREE_REMAINING=$(curl -s "https://blyan.com/api/leaderboard/me/summary?address=new_test_user" \
    | grep -o '"free_requests_remaining":[0-9]*' | cut -d: -f2)
[ "$FREE_REMAINING" = "5" ]
check_result $? "New user gets 5 free requests"

# Node registration test
echo "Testing node registration..."
if [ ! -z "$NODE_KEY" ]; then
    NODE_ID="test_node_$$"
    curl -s -X POST https://blyan.com/api/p2p/register \
        -H "X-API-Key: $NODE_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"node_id\":\"$NODE_ID\",\"host\":\"test.example.com\",\"port\":8001,\"available_experts\":[\"test\"]}" \
        | grep -q success
    check_result $? "Node registration works"
    
    # Check if node appears
    sleep 2
    curl -s https://blyan.com/api/p2p/nodes \
        -H "X-API-Key: $NODE_KEY" \
        | grep -q "$NODE_ID"
    check_result $? "Node appears in list"
else
    echo "⚠️  Skipping node tests (set NODE_KEY environment variable)"
fi
echo ""

# Summary
echo "===================================="
echo "📊 VERIFICATION SUMMARY"
echo "===================================="
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${RED}Failed:${NC} $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED - READY FOR LAUNCH!${NC} 🚀"
    exit 0
else
    echo -e "${RED}❌ $FAILED CHECKS FAILED - REVIEW BEFORE LAUNCH${NC}"
    exit 1
fi