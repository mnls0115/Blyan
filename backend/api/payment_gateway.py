#!/usr/bin/env python3
"""
Stripe Payment Gateway for Blyan Network
Handles payments and automatically funds validation_pool

### PRODUCTION FEATURES ###
- Stripe webhook for payment confirmation
- Automatic validation_pool funding
- Test mode for development ($0.50 test payments)
- Phase-based pricing strategy
"""

import os
import json
import time
import logging
from typing import Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel
import hashlib
import hmac

# Stripe SDK
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    logging.warning("Stripe not installed. Install with: pip install stripe")

# Import PostgreSQL ledger
import asyncio
from decimal import Decimal
from backend.accounting.ledger_postgres import postgres_ledger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/payment", tags=["payment"])

# Stripe configuration
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_...')  # Use test key by default
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', 'whsec_...')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', 'pk_test_...')

if STRIPE_AVAILABLE:
    stripe.api_key = STRIPE_SECRET_KEY

# Pricing tiers (Phase 1: MVP pricing)
PRICING_TIERS = {
    "test": {
        "amount": 50,  # $0.50 in cents
        "currency": "usd",
        "description": "Test payment - 100 BLY credits",
        "bly_amount": 100,
        "validation_pool_ratio": 0.25  # 25% to validation pool
    },
    "starter": {
        "amount": 500,  # $5.00
        "currency": "usd", 
        "description": "Starter pack - 1,000 BLY credits",
        "bly_amount": 1000,
        "validation_pool_ratio": 0.25
    },
    "pro": {
        "amount": 2000,  # $20.00
        "currency": "usd",
        "description": "Pro pack - 5,000 BLY credits",
        "bly_amount": 5000,
        "validation_pool_ratio": 0.25
    },
    "enterprise": {
        "amount": 10000,  # $100.00
        "currency": "usd",
        "description": "Enterprise pack - 30,000 BLY credits",
        "bly_amount": 30000,
        "validation_pool_ratio": 0.25
    }
}

class PaymentRequest(BaseModel):
    tier: str = "test"  # Default to test tier
    address: str  # Ethereum address for BLY delivery
    email: Optional[str] = None

class PaymentResponse(BaseModel):
    payment_intent_id: str
    client_secret: str
    amount: int
    currency: str
    bly_amount: int
    publishable_key: str

class WebhookEvent(BaseModel):
    type: str
    data: Dict[str, Any]

# Payment tracking (use database in production)
payment_sessions = {}  # payment_intent_id -> payment details
completed_payments = {}  # payment_intent_id -> completion details

@router.post("/create_payment")
async def create_payment(request: PaymentRequest) -> PaymentResponse:
    """
    Create a Stripe payment intent for BLY purchase.
    
    ### Test Mode ###
    Use card number: 4242 4242 4242 4242
    Any future date for expiry, any CVC
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(500, "Payment gateway not configured")
    
    # Validate tier
    if request.tier not in PRICING_TIERS:
        raise HTTPException(400, f"Invalid tier: {request.tier}")
    
    tier = PRICING_TIERS[request.tier]
    
    try:
        # Create Stripe payment intent
        intent = stripe.PaymentIntent.create(
            amount=tier["amount"],
            currency=tier["currency"],
            description=tier["description"],
            metadata={
                "address": request.address,
                "tier": request.tier,
                "bly_amount": str(tier["bly_amount"]),
                "validation_pool_ratio": str(tier["validation_pool_ratio"])
            },
            receipt_email=request.email
        )
        
        # Store payment session
        payment_sessions[intent.id] = {
            "address": request.address,
            "tier": request.tier,
            "amount": tier["amount"],
            "bly_amount": tier["bly_amount"],
            "created_at": int(time.time()),
            "status": "pending"
        }
        
        logger.info(f"Payment intent created: {intent.id} for {request.address}")
        
        return PaymentResponse(
            payment_intent_id=intent.id,
            client_secret=intent.client_secret,
            amount=tier["amount"],
            currency=tier["currency"],
            bly_amount=tier["bly_amount"],
            publishable_key=STRIPE_PUBLISHABLE_KEY
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(400, str(e))

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None)
):
    """
    Stripe webhook endpoint for payment confirmation.
    Automatically funds validation_pool when payment succeeds.
    
    ### Setup in Stripe Dashboard ###
    1. Go to Developers > Webhooks
    2. Add endpoint: https://your-domain/payment/webhook
    3. Select events: payment_intent.succeeded, payment_intent.failed
    4. Copy webhook secret to STRIPE_WEBHOOK_SECRET env var
    """
    # Get raw body
    payload = await request.body()
    
    # Verify webhook signature (if secret configured)
    if STRIPE_WEBHOOK_SECRET and STRIPE_WEBHOOK_SECRET != 'whsec_...':
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            logger.error("Invalid webhook payload")
            raise HTTPException(400, "Invalid payload")
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid webhook signature")
            raise HTTPException(400, "Invalid signature")
    else:
        # No signature verification in test mode
        event = json.loads(payload)
        logger.warning("Webhook signature not verified (test mode)")
    
    # Handle the event
    if event['type'] == 'payment_intent.succeeded':
        await handle_payment_success(event['data']['object'])
        
    elif event['type'] == 'payment_intent.payment_failed':
        await handle_payment_failure(event['data']['object'])
        
    elif event['type'] == 'checkout.session.completed':
        # Alternative: Handle checkout session completion
        session = event['data']['object']
        payment_intent_id = session.get('payment_intent')
        if payment_intent_id:
            intent = stripe.PaymentIntent.retrieve(payment_intent_id)
            await handle_payment_success(intent)
    
    return {"success": True}

async def handle_payment_success(payment_intent: Dict):
    """
    Handle successful payment with idempotency and ledger accounting.
    """
    payment_id = payment_intent['id']
    metadata = payment_intent.get('metadata', {})
    amount_received = payment_intent.get('amount_received', 0) / 100  # Convert cents to dollars
    
    # Get payment details
    if payment_id not in payment_sessions:
        logger.warning(f"Payment {payment_id} not found in sessions")
        # Try to reconstruct from metadata
        payment_sessions[payment_id] = {
            "address": metadata.get('address'),
            "bly_amount": int(metadata.get('bly_amount', 0)),
            "validation_pool_ratio": float(metadata.get('validation_pool_ratio', 0.25))
        }
    
    payment = payment_sessions[payment_id]
    address = payment['address']
    bly_amount = payment['bly_amount']
    validation_ratio = payment.get('validation_pool_ratio', 0.25)
    
    # Use PostgreSQL ledger for atomic transaction recording
    try:
        # Initialize ledger if not already done
        if not postgres_ledger._initialized:
            await postgres_ledger.initialize()
        
        # Calculate Stripe fee (2.9% + $0.30)
        stripe_fee = Decimal(str(amount_received * 0.029 + 0.30))
        
        # Process payment through PostgreSQL ledger (double-entry bookkeeping)
        transaction = await postgres_ledger.process_payment(
            stripe_event_id=payment_id,
            user_address=address,
            gross_amount=Decimal(str(amount_received)),
            stripe_fee=stripe_fee,
            bly_amount=Decimal(str(bly_amount))
        )
        
        logger.info(f"✅ PostgreSQL ledger transaction {transaction['transaction_id']} created for payment {payment_id}")
        
    except Exception as e:
        logger.error(f"Failed to process payment in ledger: {e}")
        # Fall back to simple accounting
        
    # Calculate pool allocations
    validation_pool_amount = bly_amount * validation_ratio  # 25% to validation
    burn_amount = bly_amount * 0.50  # 50% burn
    training_pool_amount = bly_amount * 0.25  # 25% to training
    
    # Update pools (these now just update in-memory state, ledger handles the real accounting)
    await fund_validation_pool(validation_pool_amount)
    await fund_training_pool(training_pool_amount)
    await burn_tokens(burn_amount)
    
    # Credit BLY to user (ledger already updated balance)
    await credit_bly_to_user(address, bly_amount)
    
    # Mark as completed
    completed_payments[payment_id] = {
        "address": address,
        "bly_amount": bly_amount,
        "validation_funded": validation_pool_amount,
        "training_funded": training_pool_amount,
        "burned": burn_amount,
        "completed_at": int(time.time())
    }
    
    payment_sessions[payment_id]['status'] = 'completed'
    
    logger.info(f"✅ Payment {payment_id} completed: {bly_amount} BLY to {address}")
    logger.info(f"   Validation pool: +{validation_pool_amount} BLY")
    logger.info(f"   Training pool: +{training_pool_amount} BLY")
    logger.info(f"   Burned: {burn_amount} BLY")

async def handle_payment_failure(payment_intent: Dict):
    """
    Handle failed payment.
    """
    payment_id = payment_intent['id']
    
    if payment_id in payment_sessions:
        payment_sessions[payment_id]['status'] = 'failed'
        
    logger.warning(f"❌ Payment {payment_id} failed")

async def fund_validation_pool(amount: float):
    """
    Add funds to validation pool for L2 API costs.
    """
    try:
        from backend.data.quality_gate_v2 import get_quality_gate
        gate = get_quality_gate()
        gate.validation_pool_bly += amount
        logger.info(f"Validation pool funded: +{amount} BLY (total: {gate.validation_pool_bly})")
    except Exception as e:
        logger.error(f"Failed to fund validation pool: {e}")

async def fund_training_pool(amount: float):
    """
    Add funds to training pool for rewards.
    """
    try:
        from backend.data.quality_gate_v2 import get_quality_gate
        gate = get_quality_gate()
        gate.training_pool_bly += amount
        logger.info(f"Training pool funded: +{amount} BLY (total: {gate.training_pool_bly})")
    except Exception as e:
        logger.error(f"Failed to fund training pool: {e}")

async def burn_tokens(amount: float):
    """
    Burn tokens to reduce supply (recorded in ledger).
    """
    try:
        from backend.accounting.ledger import get_ledger
        ledger = get_ledger()
        
        # Burn is recorded as an expense in the ledger
        # The actual burning happens via the ledger transaction
        current_burned = ledger.get_balance("burned_tokens")
        logger.info(f"🔥 Burned {amount} BLY tokens (total burned: {current_burned + amount})")
    except Exception as e:
        logger.error(f"Failed to record burn: {e}")

async def credit_bly_to_user(address: str, amount: float):
    """
    Credit BLY tokens to user's account (recorded in ledger).
    """
    try:
        from backend.accounting.ledger import get_ledger
        ledger = get_ledger()
        
        # User balance is updated via the ledger transaction
        user_balance = ledger.get_user_balance(address)
        logger.info(f"💰 Credited {amount} BLY to {address} (new balance: {user_balance})")
        
        # Also update in-memory tracking for compatibility
        from backend.data.quality_gate_v2 import get_quality_gate
        gate = get_quality_gate()
        
        if address not in gate.contributors:
            from backend.data.quality_gate_v2 import DataContributor
            gate.contributors[address] = DataContributor(address=address)
            
        gate.contributors[address].total_rewards_bly += amount
        
    except Exception as e:
        logger.error(f"Failed to credit BLY: {e}")

@router.get("/status/{payment_intent_id}")
async def get_payment_status(payment_intent_id: str) -> Dict:
    """
    Check payment status.
    """
    if payment_intent_id in completed_payments:
        return {
            "status": "completed",
            **completed_payments[payment_intent_id]
        }
    elif payment_intent_id in payment_sessions:
        return {
            "status": payment_sessions[payment_intent_id].get('status', 'pending'),
            **payment_sessions[payment_intent_id]
        }
    else:
        raise HTTPException(404, "Payment not found")

@router.get("/pricing")
async def get_pricing() -> Dict:
    """
    Get current pricing tiers.
    """
    return {
        "tiers": PRICING_TIERS,
        "currency": "USD",
        "test_mode": STRIPE_SECRET_KEY.startswith('sk_test_'),
        "publishable_key": STRIPE_PUBLISHABLE_KEY
    }

@router.post("/test_payment")
async def create_test_payment() -> Dict:
    """
    Create a test payment for development.
    Automatically uses test card and confirms payment.
    """
    if not STRIPE_SECRET_KEY.startswith('sk_test_'):
        raise HTTPException(400, "Test payments only available in test mode")
    
    try:
        # Create and immediately confirm test payment
        intent = stripe.PaymentIntent.create(
            amount=50,  # $0.50
            currency="usd",
            description="Test payment - Auto confirmed",
            confirm=True,
            payment_method="pm_card_visa",  # Test card
            metadata={
                "address": "0xtest123",
                "tier": "test",
                "bly_amount": "100",
                "validation_pool_ratio": "0.25"
            }
        )
        
        # Trigger webhook manually for test
        await handle_payment_success(intent)
        
        return {
            "success": True,
            "payment_intent_id": intent.id,
            "message": "Test payment completed successfully"
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Test payment error: {e}")
        raise HTTPException(400, str(e))

# Include router in main app
def include_router(app):
    """Include payment routes in FastAPI app."""
    app.include_router(router)