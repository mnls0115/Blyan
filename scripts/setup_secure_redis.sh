#!/bin/bash
# Setup secure Redis for production Blyan deployment

set -e

echo "🔐 Setting up secure Redis for Blyan"
echo "===================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Install Redis if not present
if ! command -v redis-server &> /dev/null; then
    echo "📦 Installing Redis..."
    apt-get update
    apt-get install -y redis-server redis-tools
fi

# Create Redis user and directories
echo "📁 Creating Redis directories..."
mkdir -p /etc/redis/certs
mkdir -p /var/log/redis
chown redis:redis /var/log/redis

# Generate passwords
echo "🔑 Generating secure passwords..."
REDIS_PASSWORD=$(openssl rand -base64 32)
API_SERVER_PASSWORD=$(openssl rand -base64 32)
GPU_NODE_PASSWORD=$(openssl rand -base64 32)
MONITORING_PASSWORD=$(openssl rand -base64 32)
BACKUP_PASSWORD=$(openssl rand -base64 32)
ADMIN_PASSWORD=$(openssl rand -base64 32)

# Generate self-signed certificates for TLS
echo "🔒 Generating TLS certificates..."
cd /etc/redis/certs

# Generate CA
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt \
    -subj "/C=US/ST=State/L=City/O=Blyan/CN=Blyan-Redis-CA"

# Generate server certificate
openssl genrsa -out redis.key 2048
openssl req -new -key redis.key -out redis.csr \
    -subj "/C=US/ST=State/L=City/O=Blyan/CN=redis.blyan.local"
openssl x509 -req -in redis.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out redis.crt -days 365 -sha256

# Generate DH params
openssl dhparam -out redis.dh 2048

# Set permissions
chmod 400 redis.key
chmod 444 redis.crt ca.crt redis.dh
chown redis:redis /etc/redis/certs/*

# Create ACL file with generated passwords
echo "📝 Creating ACL configuration..."
cat > /etc/redis/users.acl << EOF
# Redis ACL Configuration for Blyan
# Generated on $(date)

# Default user - disabled in production
user default off

# API Server user - full access to GPU node data
user api-server on >${API_SERVER_PASSWORD} ~gpu:* ~metrics:* +@all

# GPU Node user - limited to registration and heartbeat
user gpu-node on >${GPU_NODE_PASSWORD} ~gpu:* ~heartbeat:* +@read +@write +@list +@set +@hash -flushdb -flushall

# Monitoring user - read-only access
user monitoring on >${MONITORING_PASSWORD} ~* +@read +ping +info +client

# Backup user - read-only for persistence
user backup on >${BACKUP_PASSWORD} ~* +@read +bgsave +lastsave

# Admin user - full access
user admin on >${ADMIN_PASSWORD} ~* &* +@all
EOF

chmod 640 /etc/redis/users.acl
chown redis:redis /etc/redis/users.acl

# Create Redis configuration
echo "⚙️ Creating Redis configuration..."
cat > /etc/redis/redis-blyan.conf << EOF
# Redis Configuration for Blyan
# Generated on $(date)

# Basic settings
daemonize yes
pidfile /var/run/redis/redis-blyan.pid
loglevel notice
logfile /var/log/redis/redis-blyan.log
databases 16

# Network
bind 127.0.0.1 ::1
protected-mode yes
port 0
tls-port 6380
timeout 300
tcp-keepalive 60

# TLS Configuration
tls-cert-file /etc/redis/certs/redis.crt
tls-key-file /etc/redis/certs/redis.key
tls-ca-cert-file /etc/redis/certs/ca.crt
tls-dh-params-file /etc/redis/certs/redis.dh
tls-protocols "TLSv1.2 TLSv1.3"
tls-prefer-server-ciphers yes
tls-session-caching yes

# Security
requirepass ${REDIS_PASSWORD}
aclfile /etc/redis/users.acl
rename-command FLUSHDB ""
rename-command FLUSHALL ""

# Persistence
dir /var/lib/redis
dbfilename blyan.rdb
appendonly yes
appendfilename "blyan.aof"
appendfsync everysec

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 512
EOF

# Create systemd service
echo "🚀 Creating systemd service..."
cat > /etc/systemd/system/redis-blyan.service << EOF
[Unit]
Description=Redis for Blyan GPU Node Manager
After=network.target

[Service]
Type=notify
ExecStart=/usr/bin/redis-server /etc/redis/redis-blyan.conf --supervised systemd
ExecStop=/usr/bin/redis-cli -p 6380 --tls --cert /etc/redis/certs/redis.crt --key /etc/redis/certs/redis.key --cacert /etc/redis/certs/ca.crt shutdown
TimeoutStopSec=0
Restart=on-failure
User=redis
Group=redis
RuntimeDirectory=redis
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
systemctl daemon-reload
systemctl enable redis-blyan
systemctl start redis-blyan

# Wait for Redis to start
sleep 2

# Test connection
echo "🧪 Testing Redis connection..."
redis-cli -p 6380 --tls \
    --cert /etc/redis/certs/redis.crt \
    --key /etc/redis/certs/redis.key \
    --cacert /etc/redis/certs/ca.crt \
    --user admin --pass ${ADMIN_PASSWORD} \
    ping

# Save credentials
echo "💾 Saving credentials..."
cat > /root/redis-blyan-credentials.txt << EOF
# Redis Credentials for Blyan
# Generated on $(date)
# KEEP THIS FILE SECURE!

Redis URL (with TLS):
rediss://api-server:${API_SERVER_PASSWORD}@localhost:6380/0

Redis URL (without TLS - local only):
redis://api-server:${API_SERVER_PASSWORD}@localhost:6379/0

User Passwords:
- api-server: ${API_SERVER_PASSWORD}
- gpu-node: ${GPU_NODE_PASSWORD}
- monitoring: ${MONITORING_PASSWORD}
- backup: ${BACKUP_PASSWORD}
- admin: ${ADMIN_PASSWORD}
- default Redis password: ${REDIS_PASSWORD}

Environment variables for .env:
REDIS_URL=rediss://api-server:${API_SERVER_PASSWORD}@localhost:6380/0
REDIS_USERNAME=api-server
REDIS_PASSWORD=${API_SERVER_PASSWORD}
REDIS_SSL=true
REDIS_SSL_CERTFILE=/etc/redis/certs/redis.crt
REDIS_SSL_KEYFILE=/etc/redis/certs/redis.key
REDIS_SSL_CA_CERTS=/etc/redis/certs/ca.crt
EOF

chmod 600 /root/redis-blyan-credentials.txt

echo ""
echo "✅ Secure Redis setup complete!"
echo ""
echo "📋 Important information:"
echo "  - Redis is listening on port 6380 (TLS only)"
echo "  - Credentials saved to: /root/redis-blyan-credentials.txt"
echo "  - ACL configuration: /etc/redis/users.acl"
echo "  - TLS certificates: /etc/redis/certs/"
echo "  - Service name: redis-blyan"
echo ""
echo "🔧 To manage Redis:"
echo "  systemctl status redis-blyan"
echo "  systemctl restart redis-blyan"
echo "  journalctl -u redis-blyan -f"
echo ""
echo "🔌 To connect:"
echo "  redis-cli -p 6380 --tls --cert /etc/redis/certs/redis.crt --key /etc/redis/certs/redis.key --cacert /etc/redis/certs/ca.crt --user admin --pass <password>"
echo ""
echo "⚠️ IMPORTANT: Copy the credentials from /root/redis-blyan-credentials.txt to your .env file!"