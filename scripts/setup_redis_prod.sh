#!/bin/bash
# Redis Setup Script for Production

echo "🔴 Redis Production Setup"
echo "========================="

# Configure Redis
echo "📝 Configuring Redis..."
cat > /tmp/redis_prod.conf << 'REDIS'
# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
dbfilename dump.rdb
dir /var/lib/redis

# ACL
aclfile /etc/redis/users.acl
requirepass your_redis_master_password_here

# Security
protected-mode yes
bind 127.0.0.1
port 6379

# Performance
tcp-backlog 511
tcp-keepalive 300
timeout 0

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
REDIS

# Copy ACL file
echo "🔐 Setting up ACL..."
sudo cp /root/blyan/config/redis-users.acl /etc/redis/users.acl
sudo chmod 640 /etc/redis/users.acl
sudo chown redis:redis /etc/redis/users.acl

# Apply configuration
echo "⚙️ Applying configuration..."
sudo cp /tmp/redis_prod.conf /etc/redis/redis.conf
sudo systemctl restart redis

# Verify setup
echo ""
echo "🔍 Verifying Redis setup..."
echo "ACL users:"
redis-cli -a your_redis_master_password_here ACL LIST

echo ""
echo "Memory policy:"
redis-cli -a your_redis_master_password_here CONFIG GET maxmemory-policy

echo ""
echo "Persistence settings:"
redis-cli -a your_redis_master_password_here CONFIG GET save

echo ""
echo "✅ Redis setup complete!"
echo ""
echo "⚠️  IMPORTANT: Update passwords in:"
echo "  - /etc/redis/users.acl"
echo "  - /etc/redis/redis.conf (requirepass)"
echo "  - Application environment variables"