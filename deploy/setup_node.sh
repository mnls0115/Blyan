#!/bin/bash
# Blyan Network P2P Node Setup Script
# For Ubuntu 22.04+ and compatible systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BLYAN_USER="blyan"
BLYAN_DIR="/opt/blyan"
PYTHON_VERSION="3.10"
NODE_TYPE="${1:-FULL}"  # SEED, FULL, or LIGHT
NODE_ID="${2:-$(hostname)}"
EXTERNAL_IP="${3:-}"

echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Blyan Network P2P Node Setup Script      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "Node Type: $NODE_TYPE"
echo "Node ID: $NODE_ID"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
apt update && apt upgrade -y

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
apt install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    build-essential \
    git \
    curl \
    wget \
    htop \
    iotop \
    net-tools \
    ufw \
    fail2ban \
    nginx \
    certbot \
    python3-certbot-nginx

# Create blyan user
if ! id "$BLYAN_USER" &>/dev/null; then
    echo -e "${YELLOW}Creating blyan user...${NC}"
    useradd -r -m -d /home/$BLYAN_USER -s /bin/bash $BLYAN_USER
fi

# Setup directory structure
echo -e "${YELLOW}Setting up directory structure...${NC}"
mkdir -p $BLYAN_DIR/{data,logs,config}
chown -R $BLYAN_USER:$BLYAN_USER $BLYAN_DIR

# Clone or update repository
if [ -d "$BLYAN_DIR/.git" ]; then
    echo -e "${YELLOW}Updating repository...${NC}"
    cd $BLYAN_DIR
    sudo -u $BLYAN_USER git pull origin main
else
    echo -e "${YELLOW}Cloning repository...${NC}"
    cd /opt
    git clone https://github.com/mnls0115/Blyan.git blyan
    chown -R $BLYAN_USER:$BLYAN_USER $BLYAN_DIR
fi

# Setup Python virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
cd $BLYAN_DIR
sudo -u $BLYAN_USER python${PYTHON_VERSION} -m venv .venv

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
sudo -u $BLYAN_USER .venv/bin/pip install --upgrade pip
sudo -u $BLYAN_USER .venv/bin/pip install -r requirements.txt

# Generate node configuration
echo -e "${YELLOW}Generating node configuration...${NC}"
cat > $BLYAN_DIR/config/node.yaml <<EOF
# Blyan Network Node Configuration
# Auto-generated on $(date)

node:
  id: "$NODE_ID"
  role: "$NODE_TYPE"
  data_dir: "$BLYAN_DIR/data"

network:
  listen_addr: "0.0.0.0:4321"
  external_addr: "${EXTERNAL_IP:-}:4321"
  max_peers: $([ "$NODE_TYPE" = "SEED" ] && echo "256" || echo "64")

bootstrap:
  nodes:
    - "seed1.blyan.com:4321"
    - "seed2.blyan.com:4321"
    - "165.227.221.225:4321"
  
  dht:
    enabled: true
    bootstrap_nodes:
      - "dht.blyan.com:4322"

p2p:
  protocol_version: "1.0"
  sync:
    headers_batch_size: 100
    blocks_batch_size: 32
    parallel_downloads: 4

security:
  key_file: "$BLYAN_DIR/data/node.key"
  ban:
    score_threshold: 0
    ban_duration: 3600

api:
  enabled: true
  listen_addr: "127.0.0.1:8080"

metrics:
  enabled: true
  listen_addr: "127.0.0.1:9090"

logging:
  level: "INFO"
  format: "json"
  outputs:
    - type: "file"
      path: "$BLYAN_DIR/logs/node.log"
      level: "INFO"
      max_size: 104857600
      max_backups: 10
EOF

chown $BLYAN_USER:$BLYAN_USER $BLYAN_DIR/config/node.yaml

# Setup systemd service
echo -e "${YELLOW}Setting up systemd service...${NC}"
cat > /etc/systemd/system/blyan-node.service <<EOF
[Unit]
Description=Blyan Network P2P Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$BLYAN_USER
Group=$BLYAN_USER
WorkingDirectory=$BLYAN_DIR

Environment="NODE_ENV=production"
Environment="PYTHONPATH=$BLYAN_DIR"

LimitNOFILE=65536
LimitNPROC=4096

Restart=always
RestartSec=10

ExecStart=$BLYAN_DIR/.venv/bin/python -m backend.p2p.node_runner --config $BLYAN_DIR/config/node.yaml
ExecStop=/bin/kill -TERM \$MAINPID

StandardOutput=journal
StandardError=journal
SyslogIdentifier=blyan-node

[Install]
WantedBy=multi-user.target
EOF

# Setup firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw allow 22/tcp comment 'SSH'
ufw allow 4321/tcp comment 'Blyan P2P'
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'

# Enable UFW if not already enabled
if ! ufw status | grep -q "Status: active"; then
    ufw --force enable
fi

# Setup nginx reverse proxy for API
echo -e "${YELLOW}Setting up nginx...${NC}"
cat > /etc/nginx/sites-available/blyan-api <<EOF
server {
    listen 80;
    server_name _;

    location /api/ {
        proxy_pass http://127.0.0.1:8080/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /metrics {
        proxy_pass http://127.0.0.1:9090/metrics;
        allow 127.0.0.1;
        deny all;
    }
}
EOF

ln -sf /etc/nginx/sites-available/blyan-api /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# Setup log rotation
echo -e "${YELLOW}Setting up log rotation...${NC}"
cat > /etc/logrotate.d/blyan <<EOF
$BLYAN_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $BLYAN_USER $BLYAN_USER
    sharedscripts
    postrotate
        systemctl reload blyan-node
    endscript
}
EOF

# Enable and start service
echo -e "${YELLOW}Starting Blyan node service...${NC}"
systemctl daemon-reload
systemctl enable blyan-node
systemctl start blyan-node

# Wait for service to start
sleep 5

# Check service status
if systemctl is-active --quiet blyan-node; then
    echo -e "${GREEN}✅ Blyan node is running!${NC}"
    systemctl status blyan-node --no-pager
else
    echo -e "${RED}❌ Failed to start Blyan node${NC}"
    journalctl -u blyan-node -n 50 --no-pager
    exit 1
fi

# Display connection information
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Setup Complete!                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "Node ID: $NODE_ID"
echo "Node Type: $NODE_TYPE"
echo "P2P Port: 4321"
echo "API Port: 8080 (local)"
echo "Metrics Port: 9090 (local)"
echo ""
echo "Useful commands:"
echo "  systemctl status blyan-node    # Check status"
echo "  systemctl restart blyan-node   # Restart node"
echo "  journalctl -u blyan-node -f    # View logs"
echo "  curl http://localhost:8080/health  # Check health"
echo ""

# Test API endpoint
echo -e "${YELLOW}Testing API endpoint...${NC}"
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ API is responding${NC}"
else
    echo -e "${YELLOW}⚠️  API not yet ready, check logs${NC}"
fi

echo ""
echo -e "${GREEN}Node setup complete!${NC}"