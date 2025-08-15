# Two-Chain Architecture Makefile

.PHONY: help install test demo localnet clean

help:
	@echo "Two-Chain Architecture Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run integration tests"
	@echo "  make demo       - Run demo script"
	@echo "  make localnet   - Start local testnet"
	@echo "  make clean      - Clean build artifacts"

install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio

test:
	pytest tests/test_two_chain.py -v

demo:
	python demo_two_chain.py

localnet:
	@echo "Starting local two-chain testnet..."
	@echo "TX-Chain: http://localhost:8545"
	@echo "PoL-Chain: http://localhost:8546"
	@echo "Relayer: http://localhost:8547"
	python -m blockchain.tx_chain.node --port 8545 &
	python -m blockchain.pol_chain.node --port 8546 &
	python -m blockchain.relayer.service --tx-url http://localhost:8545 --pol-url http://localhost:8546 &

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete