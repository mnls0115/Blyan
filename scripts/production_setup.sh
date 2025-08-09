#!/bin/bash

# Blyan Network Production Setup Script
# Complete setup for Week 1 production transition

set -e  # Exit on error

echo "🚀 Blyan Network Production Setup"
echo "=================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (needed for some operations)
if [ "$EUID" -eq 0 ]; then 
   echo -e "${YELLOW}⚠️  Running as root - be careful!${NC}"
fi

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 1. Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    print_error "Docker not installed"
    exit 1
else
    print_status "Docker installed"
fi

if ! command_exists docker-compose; then
    print_error "Docker Compose not installed"
    exit 1
else
    print_status "Docker Compose installed"
fi

if ! command_exists python3; then
    print_error "Python 3 not installed"
    exit 1
else
    print_status "Python 3 installed"
fi

if ! command_exists redis-cli; then
    print_warning "Redis CLI not installed (optional)"
fi

if ! command_exists psql; then
    print_warning "PostgreSQL client not installed (optional)"
fi

echo ""

# 2. Create necessary directories
echo "📁 Creating directory structure..."

directories=(
    "models"
    "data"
    "logs"
    "migrations"
    "ssl"
    "backups"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created $dir/"
    else
        echo "   $dir/ already exists"
    fi
done

echo ""

# 3. Check/Create environment file
echo "🔐 Setting up environment configuration..."

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        print_status "Created .env from .env.example"
        print_warning "IMPORTANT: Edit .env and set all passwords!"
    else
        print_error ".env.example not found"
        exit 1
    fi
else
    print_status ".env already exists"
fi

# Check for production env file
if [ ! -f .env.production ]; then
    print_warning ".env.production not found - using .env"
else
    print_status ".env.production exists"
fi

echo ""

# 4. Prepare Teacher Model
echo "🤖 Preparing Teacher Model..."

if [ -f "models/teacher_v17-int8.safetensors" ]; then
    print_status "Teacher model already exists"
    python3 scripts/prepare_teacher_model.py --verify-only
else
    print_warning "Teacher model not found, creating..."
    python3 scripts/prepare_teacher_model.py --output models/teacher_v17-int8.safetensors --size-mb 100
fi

echo ""

# 5. Setup Redis TLS (optional)
echo "🔒 Setting up Redis security..."

if [ "$1" == "--with-tls" ]; then
    if [ ! -d "/etc/redis/certs" ]; then
        print_warning "Setting up Redis TLS certificates..."
        sudo bash scripts/setup_redis_tls.sh
    else
        print_status "Redis TLS certificates already configured"
    fi
else
    print_warning "Skipping TLS setup (use --with-tls to enable)"
fi

# Check Redis configuration
if [ -f "redis.conf" ]; then
    print_status "Redis configuration found"
else
    print_error "redis.conf not found!"
fi

echo ""

# 6. Initialize PostgreSQL
echo "💾 Preparing PostgreSQL..."

if [ -f "migrations/001_create_ledger.sql" ]; then
    print_status "Ledger migration script found"
else
    print_error "Ledger migration script not found!"
fi

if [ -f "init.sql" ]; then
    print_status "Database init script found"
else
    print_error "Database init script not found!"
fi

echo ""

# 7. Check API keys
echo "🔑 Checking External API configuration..."

source .env 2>/dev/null || source .env.production 2>/dev/null || true

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" == "sk-..." ]; then
    print_warning "OpenAI API key not configured"
else
    print_status "OpenAI API key configured"
fi

if [ -z "$PERSPECTIVE_API_KEY" ] || [ "$PERSPECTIVE_API_KEY" == "..." ]; then
    print_warning "Perspective API key not configured"
else
    print_status "Perspective API key configured"
fi

if [ -z "$STRIPE_SECRET_KEY" ] || [ "$STRIPE_SECRET_KEY" == "sk_live_..." ]; then
    print_warning "Stripe API key not configured"
else
    print_status "Stripe API key configured"
fi

echo ""

# 8. Docker services check
echo "🐳 Checking Docker services..."

# Check if services are running
if docker-compose ps 2>/dev/null | grep -q "Up"; then
    print_status "Docker services are running"
    echo ""
    docker-compose ps
else
    print_warning "Docker services not running"
    echo ""
    read -p "Start Docker services now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose up -d
        sleep 5
        docker-compose ps
    fi
fi

echo ""

# 9. Database connectivity test
echo "🔍 Testing database connections..."

# Test Redis
if docker-compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    print_status "Redis is responding"
else
    print_error "Redis not responding"
fi

# Test PostgreSQL
if docker-compose exec -T postgres pg_isready 2>/dev/null | grep -q "accepting connections"; then
    print_status "PostgreSQL is ready"
else
    print_error "PostgreSQL not ready"
fi

echo ""

# 10. API health check
echo "❤️  Checking API health..."

API_URL="http://localhost:8000/health"

if curl -s $API_URL > /dev/null 2>&1; then
    print_status "API is responding"
    
    # Check teacher model endpoint
    if curl -s $API_URL/teacher | grep -q "healthy"; then
        print_status "Teacher model endpoint healthy"
    else
        print_warning "Teacher model endpoint not healthy"
    fi
else
    print_warning "API not responding (may need to start)"
fi

echo ""

# 11. Generate summary report
echo "📊 Production Readiness Report"
echo "=============================="
echo ""

readiness_score=0
max_score=10

# Check critical components
components=(
    "Docker:docker"
    "Redis:redis-cli"
    "PostgreSQL:psql"
    "Teacher Model:models/teacher_v17-int8.safetensors"
    "Ledger Migration:migrations/001_create_ledger.sql"
    "Environment Config:.env"
    "Redis Config:redis.conf"
)

for component in "${components[@]}"; do
    IFS=':' read -r name check <<< "$component"
    if [ -f "$check" ] || command_exists "$check" 2>/dev/null; then
        echo "✅ $name"
        ((readiness_score++))
    else
        echo "❌ $name"
    fi
done

# Check API keys
if [ ! -z "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "sk-..." ]; then
    echo "✅ OpenAI API"
    ((readiness_score++))
else
    echo "⚠️  OpenAI API (optional)"
fi

if [ ! -z "$STRIPE_SECRET_KEY" ] && [ "$STRIPE_SECRET_KEY" != "sk_live_..." ]; then
    echo "✅ Stripe API"
    ((readiness_score++))
else
    echo "❌ Stripe API (required)"
fi

echo ""
echo "Production Readiness: $readiness_score/$max_score"

if [ $readiness_score -ge 8 ]; then
    print_status "System is READY for production! 🎉"
elif [ $readiness_score -ge 6 ]; then
    print_warning "System is PARTIALLY ready (configure API keys)"
else
    print_error "System is NOT ready for production"
fi

echo ""
echo "📝 Next Steps:"
echo "1. Edit .env file with production passwords"
echo "2. Configure external API keys (OpenAI, Perspective, Stripe)"
echo "3. Run: docker-compose up -d"
echo "4. Monitor logs: docker-compose logs -f"
echo "5. Check health: curl http://localhost:8000/health"
echo ""
echo "For full production deployment to Digital Ocean:"
echo "   ./deploy_digitalocean.sh"
echo ""