#!/usr/bin/env python3
"""Test A2A connection to green agent"""
import asyncio
import httpx
import sys

async def test_connection():
    """Test basic connection and A2A client"""
    base_url = "http://green-agent:9008"

    print(f"Testing connection to {base_url}...")

    # Test 1: Basic HTTP GET
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base_url}/health")
            print(f"✓ Health check: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

    # Test 2: Agent card
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            from a2a.client import A2ACardResolver
            resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
            card = await resolver.get_agent_card()
            print(f"✓ Agent card retrieved: {card.name}")
    except Exception as e:
        print(f"✗ Agent card failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Try to create client
    try:
        async with httpx.AsyncClient(timeout=10) as httpx_client:
            from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(httpx_client=httpx_client, streaming=True)
            factory = ClientFactory(config)
            client = factory.create(agent_card)
            print(f"✓ A2A client created successfully")
            print(f"  Transport: {agent_card.preferredTransport}")
            print(f"  URL: {agent_card.url}")
    except Exception as e:
        print(f"✗ Client creation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ All connection tests passed!")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    sys.exit(0 if result else 1)
