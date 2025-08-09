#!/bin/bash

# Redis TLS Certificate Setup Script
# Generates self-signed certificates for Redis TLS

set -e

echo "🔐 Setting up Redis TLS certificates..."

# Create certificate directory
CERT_DIR="/etc/redis/certs"
mkdir -p $CERT_DIR

# Certificate parameters
COUNTRY="US"
STATE="California"
CITY="San Francisco"
ORG="Blyan Network"
CN="redis.blyan.local"
DAYS=3650  # 10 years

# Generate private key
echo "Generating private key..."
openssl genrsa -out $CERT_DIR/redis.key 4096

# Generate certificate signing request
echo "Generating CSR..."
openssl req -new -key $CERT_DIR/redis.key -out $CERT_DIR/redis.csr \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=$CN"

# Generate self-signed certificate
echo "Generating certificate..."
openssl x509 -req -days $DAYS -in $CERT_DIR/redis.csr \
    -signkey $CERT_DIR/redis.key -out $CERT_DIR/redis.crt

# Generate DH parameters for perfect forward secrecy
echo "Generating DH parameters (this may take a while)..."
openssl dhparam -out $CERT_DIR/redis.dh 2048

# Create CA certificate (for client verification)
echo "Creating CA certificate..."
openssl genrsa -out $CERT_DIR/ca.key 4096
openssl req -new -x509 -days $DAYS -key $CERT_DIR/ca.key -out $CERT_DIR/ca.crt \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=Blyan CA"

# Generate client certificate for API service
echo "Generating client certificate..."
openssl genrsa -out $CERT_DIR/client.key 4096
openssl req -new -key $CERT_DIR/client.key -out $CERT_DIR/client.csr \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/CN=api-client"
openssl x509 -req -days $DAYS -in $CERT_DIR/client.csr \
    -CA $CERT_DIR/ca.crt -CAkey $CERT_DIR/ca.key -CAcreateserial \
    -out $CERT_DIR/client.crt

# Set proper permissions
chmod 600 $CERT_DIR/*.key
chmod 644 $CERT_DIR/*.crt
chmod 644 $CERT_DIR/*.dh

# Create Redis SSL client configuration
cat > $CERT_DIR/redis-cli.conf << EOF
tls-cert-file $CERT_DIR/client.crt
tls-key-file $CERT_DIR/client.key
tls-ca-cert-file $CERT_DIR/ca.crt
EOF

echo "✅ Redis TLS certificates generated successfully!"
echo ""
echo "📁 Certificates location: $CERT_DIR"
echo ""
echo "🔧 To connect with redis-cli using TLS:"
echo "   redis-cli --tls --cacert $CERT_DIR/ca.crt \\"
echo "            --cert $CERT_DIR/client.crt \\"
echo "            --key $CERT_DIR/client.key \\"
echo "            -h localhost -p 6379"
echo ""
echo "🐳 For Docker, mount the certificates:"
echo "   volumes:"
echo "     - $CERT_DIR:/etc/redis/certs:ro"