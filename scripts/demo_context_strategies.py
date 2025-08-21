#!/usr/bin/env python3
"""
Demo: Context Management Strategies
Shows different approaches to handling conversation context
"""

import sys
import time
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.context import ConversationManager, ContextStrategy

def simulate_conversation(manager: ConversationManager, turns: list) -> None:
    """Simulate a conversation with given turns"""
    conversation_id = "demo_conv_001"
    manager.start_conversation(conversation_id)
    
    print(f"\n🎯 Strategy: {manager.strategy.value}")
    print("=" * 50)
    
    total_tokens_sent = 0
    
    for i, (role, content) in enumerate(turns):
        print(f"\n[Turn {i+1}] {role}: {content[:50]}{'...' if len(content) > 50 else ''}")
        
        # Add turn to conversation
        manager.add_turn(conversation_id, role, content)
        
        # Get context that would be sent to model
        context = manager.get_context_for_inference(conversation_id)
        
        tokens_this_request = context['token_estimate']
        total_tokens_sent += tokens_this_request
        
        print(f"  📤 Tokens sent: {tokens_this_request}")
        print(f"  📊 Strategy used: {context['strategy_used']}")
        print(f"  🧠 Has KV cache: {'Yes' if context['kv_cache'] else 'No'}")
        print(f"  💰 Total tokens sent so far: {total_tokens_sent}")
        
        # Simulate KV cache storage for relevant strategies
        if manager.strategy in [ContextStrategy.KV_CACHE, ContextStrategy.HYBRID]:
            if role == "assistant":
                # Mock storing KV cache after model response
                mock_key_states = torch.randn(1, 8, 512, 64)  # Mock attention states
                mock_value_states = torch.randn(1, 8, 512, 64)
                manager.store_kv_cache(
                    conversation_id, 
                    mock_key_states, 
                    mock_value_states, 
                    len(manager.conversations[conversation_id])
                )
    
    # Final stats
    stats = manager.get_conversation_stats(conversation_id)
    print(f"\n📈 Final Stats:")
    print(f"  - Total turns: {stats['turn_count']}")
    print(f"  - Total conversation tokens: {stats['total_tokens']}")
    print(f"  - Total tokens sent to model: {total_tokens_sent}")
    
    if stats['total_tokens'] > 0:
        efficiency = (1 - total_tokens_sent / (stats['total_tokens'] * (stats['turn_count'] // 2))) * 100
        print(f"  - Token efficiency: {efficiency:.1f}% savings")

def main():
    print("🔄 Context Management Strategy Comparison")
    print("Simulating a long conversation with different strategies")
    
    # Sample conversation turns
    conversation_turns = [
        ("user", "Hello, I'm working on a Python project and need help with async programming."),
        ("assistant", "I'd be happy to help you with async programming in Python! Async programming allows you to write concurrent code using the async/await syntax. What specific aspect would you like to learn about?"),
        ("user", "Can you explain how asyncio.gather() works?"),
        ("assistant", "asyncio.gather() is a function that runs multiple coroutines concurrently and waits for all of them to complete. It takes multiple awaitable objects as arguments and returns their results as a list in the same order. Here's a simple example: await asyncio.gather(fetch_data(), process_data(), save_data())"),
        ("user", "That's helpful! What about error handling in async functions?"),
        ("assistant", "Error handling in async functions works similarly to synchronous code, but with some important considerations. You can use try/except blocks around await expressions. When using asyncio.gather(), if any coroutine raises an exception, gather() will also raise that exception, cancelling other pending coroutines unless you use return_exceptions=True."),
        ("user", "How do I handle timeouts in async operations?"),
        ("assistant", "You can handle timeouts using asyncio.wait_for(). It takes a coroutine and a timeout value in seconds. If the operation doesn't complete within the timeout, it raises asyncio.TimeoutError. Example: await asyncio.wait_for(slow_operation(), timeout=5.0)"),
        ("user", "Perfect! Can you show me a complete example combining these concepts?"),
        ("assistant", "Here's a complete example combining async programming concepts:\n\nimport asyncio\nimport aiohttp\n\nasync def fetch_url(session, url):\n    try:\n        async with session.get(url) as response:\n            return await response.text()\n    except Exception as e:\n        return f'Error: {e}'\n\nasync def main():\n    urls = ['http://example.com', 'http://google.com']\n    \n    async with aiohttp.ClientSession() as session:\n        try:\n            results = await asyncio.wait_for(\n                asyncio.gather(*[fetch_url(session, url) for url in urls]),\n                timeout=10.0\n            )\n            for url, result in zip(urls, results):\n                print(f'{url}: {len(result)} chars')\n        except asyncio.TimeoutError:\n            print('Operation timed out!')\n\nif __name__ == '__main__':\n    asyncio.run(main())"),
    ]
    
    strategies = [
        ContextStrategy.FULL_CONTEXT,
        ContextStrategy.SLIDING_WINDOW,
        ContextStrategy.KV_CACHE,
        ContextStrategy.HYBRID
    ]
    
    results = {}
    
    for strategy in strategies:
        manager = ConversationManager(
            strategy=strategy,
            sliding_window_size=4,  # Keep last 4 turns for sliding window
            kv_cache_ttl=3600
        )
        
        print(f"\n" + "="*60)
        simulate_conversation(manager, conversation_turns)
        
        # Store results for comparison
        conv_id = list(manager.conversations.keys())[0]
        stats = manager.get_conversation_stats(conv_id)
        results[strategy.value] = stats
    
    # Summary comparison
    print(f"\n" + "="*60)
    print("📊 STRATEGY COMPARISON SUMMARY")
    print("="*60)
    
    print(f"{'Strategy':<15} {'Turns':<6} {'Total Tokens':<12} {'Efficiency':<12}")
    print("-" * 50)
    
    for strategy_name, stats in results.items():
        print(f"{strategy_name:<15} {stats['turn_count']:<6} {stats['total_tokens']:<12} {'Variable':<12}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"• Development/Testing: full_context (simple, stateless)")
    print(f"• Short conversations: sliding (memory efficient)")  
    print(f"• Production systems: hybrid (best balance)")
    print(f"• Long conversations: kv_cache (maximum efficiency)")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"• Full context: Simple but expensive for long conversations")
    print(f"• KV cache: 10-100x faster for long conversations, complex state management")
    print(f"• Hybrid: Production-ready balance of performance and reliability")
    print(f"• Sliding window: Good middle ground for memory-constrained environments")

if __name__ == "__main__":
    main()