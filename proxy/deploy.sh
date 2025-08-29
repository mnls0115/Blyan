#!/bin/bash
# Blyan API Proxy Deployment Script
# Deploy the dynamic reverse proxy on DigitalOcean or any Linux server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROXY_USER="blyan"
PROXY_DIR="/opt/blyan/proxy"
NGINX_SITES="/etc/nginx/sites-available"
SYSTEMD_DIR="/etc/systemd/system"

echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Blyan API Proxy Deployment              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run as root${NC}"
   exit 1
fi

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Install dependencies
echo -e "${YELLOW}[1/7] Installing system dependencies...${NC}"
apt-get update
apt-get install -y nginx nodejs npm git curl wget certbot python3-certbot-nginx

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo -e "${YELLOW}Node.js version too old. Installing Node.js 18...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs
fi

# Step 2: Create user and directories
echo -e "${YELLOW}[2/7] Creating user and directories...${NC}"
if ! id "$PROXY_USER" &>/dev/null; then
    useradd -r -s /bin/bash -m -d /home/$PROXY_USER $PROXY_USER
    echo -e "${GREEN}Created user: $PROXY_USER${NC}"
fi

mkdir -p $PROXY_DIR
mkdir -p $PROXY_DIR/data
mkdir -p /var/log/blyan

# Step 3: Copy proxy files
echo -e "${YELLOW}[3/7] Installing proxy application...${NC}"
cp -r ./* $PROXY_DIR/
chown -R $PROXY_USER:$PROXY_USER $PROXY_DIR
chmod 755 $PROXY_DIR/server.js

# Install Node.js dependencies
cd $PROXY_DIR
sudo -u $PROXY_USER npm install --production

# Step 4: Configure environment
echo -e "${YELLOW}[4/7] Configuring environment...${NC}"
if [ ! -f "$PROXY_DIR/.env" ]; then
    cp $PROXY_DIR/.env.example $PROXY_DIR/.env
    
    # Generate admin token
    ADMIN_TOKEN=$(openssl rand -hex 32)
    sed -i "s/ADMIN_TOKEN=/ADMIN_TOKEN=$ADMIN_TOKEN/" $PROXY_DIR/.env
    
    echo -e "${GREEN}Generated admin token: $ADMIN_TOKEN${NC}"
    echo -e "${YELLOW}Save this token! You'll need it for admin operations.${NC}"
    
    # Ask for initial nodes
    echo
    read -p "Enter first GPU node URL (e.g., https://gpu.example.com:8000): " NODE1_URL
    if [ ! -z "$NODE1_URL" ]; then
        echo "NODE_0=primary|$NODE1_URL" >> $PROXY_DIR/.env
    fi
    
    read -p "Enter second GPU node URL (optional, press Enter to skip): " NODE2_URL
    if [ ! -z "$NODE2_URL" ]; then
        echo "NODE_1=secondary|$NODE2_URL" >> $PROXY_DIR/.env
    fi
fi

chown $PROXY_USER:$PROXY_USER $PROXY_DIR/.env
chmod 600 $PROXY_DIR/.env

# Step 5: Install systemd service
echo -e "${YELLOW}[5/7] Installing systemd service...${NC}"
cp $PROXY_DIR/systemd/blyan-proxy.service $SYSTEMD_DIR/
systemctl daemon-reload
systemctl enable blyan-proxy
systemctl start blyan-proxy

# Check if service is running
sleep 2
if systemctl is-active --quiet blyan-proxy; then
    echo -e "${GREEN}✓ Proxy service started successfully${NC}"
else
    echo -e "${RED}✗ Failed to start proxy service${NC}"
    journalctl -u blyan-proxy -n 20
    exit 1
fi

# Step 6: Configure Nginx
echo -e "${YELLOW}[6/7] Configuring Nginx...${NC}"

# Get domain name
read -p "Enter your domain name (e.g., blyan.com): " DOMAIN
if [ -z "$DOMAIN" ]; then
    DOMAIN="blyan.com"
fi

# Update Nginx config with domain
sed -i "s/blyan.com/$DOMAIN/g" $PROXY_DIR/nginx/blyan-proxy.conf

# Copy Nginx configuration
cp $PROXY_DIR/nginx/blyan-proxy.conf $NGINX_SITES/
ln -sf $NGINX_SITES/blyan-proxy.conf /etc/nginx/sites-enabled/

# Remove default site if exists
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
if nginx -t; then
    systemctl reload nginx
    echo -e "${GREEN}✓ Nginx configured successfully${NC}"
else
    echo -e "${RED}✗ Nginx configuration error${NC}"
    exit 1
fi

# Step 7: Setup SSL with Let's Encrypt
echo -e "${YELLOW}[7/7] Setting up SSL certificate...${NC}"
read -p "Setup SSL certificate with Let's Encrypt? (y/n): " SETUP_SSL

if [ "$SETUP_SSL" = "y" ]; then
    read -p "Enter email for Let's Encrypt: " LE_EMAIL
    certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos -m $LE_EMAIL
    
    # Setup auto-renewal
    systemctl enable certbot.timer
    systemctl start certbot.timer
    
    echo -e "${GREEN}✓ SSL certificate installed${NC}"
fi

# Final status check
echo
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Deployment Complete!                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo
echo "Proxy Status:"
systemctl status blyan-proxy --no-pager | head -n 5
echo
echo "Test endpoints:"
echo "  Health: http://localhost:9000/health"
echo "  Status: http://localhost:9000/_status"
echo "  Public: https://$DOMAIN/_status"
echo
echo "Admin token: $ADMIN_TOKEN"
echo
echo "Add nodes dynamically:"
echo "  curl -X POST http://localhost:9000/_admin/nodes \\"
echo "    -H 'X-Admin-Token: $ADMIN_TOKEN' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"name\": \"gpu-node\", \"baseURL\": \"https://example.com:8000\"}'"
echo
echo "View logs:"
echo "  journalctl -u blyan-proxy -f"
echo