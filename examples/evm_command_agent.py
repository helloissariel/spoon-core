import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from spoon_ai.agents import SpoonReactAI
from spoon_ai.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

@dataclass
class NetworkConfig:
    """Network configuration for EVM chains."""
    name: str
    chain_id: int
    rpc_url: str
    native_token: str
    explorer_url: str
    gas_level: str
    is_testnet: bool = False

class SpoonEVMCommandAgent:
    """
    Advanced EVM command parser and execution agent.

    This agent integrates natural language processing with comprehensive EVM operations,
    providing a user-friendly interface for blockchain interactions across multiple networks.
    """

    name: str = "SpoonEVMCommandAgent"
    system_prompt: str = """
You are SpoonOS EVM Command Agent, an intelligent blockchain assistant that can understand
natural language commands and execute various EVM operations safely and efficiently.

Your capabilities include:
- Balance queries for ETH and ERC20 tokens
- Native ETH and ERC20 token transfers
- Token swaps via DEX aggregators
- Cross-chain bridging operations
- Real-time price quotes and analysis
- Multi-network support (Ethereum, Base, Polygon, etc.)
- Safety confirmations and transaction monitoring

You support both English and Chinese commands and always prioritize user safety by:
1. Clearly explaining what operations will be performed
2. Requesting confirmation for transactions
3. Providing detailed feedback with transaction hashes
4. Offering help and examples when commands are unclear

Parse user commands intelligently and execute the appropriate EVM operations.
    """

    def __init__(self, **kwargs):
        log_level = logging.DEBUG if kwargs.get('debug', False) else logging.WARNING
        logging.basicConfig(level=log_level)

        # Import EVM tools
        from spoon_toolkits.crypto.evm import (
            EvmTransferTool,
            EvmSwapTool,
            EvmBridgeTool,
            EvmErc20TransferTool,
            EvmBalanceTool,
            EvmSwapQuoteTool,
        )

        # Create tool manager with EVM tools
        evm_tools = [
            EvmBalanceTool(),
            EvmTransferTool(),
            EvmErc20TransferTool(),
            EvmSwapTool(),
            EvmSwapQuoteTool(),
            EvmBridgeTool(),
        ]
        evm_tool_manager = ToolManager(evm_tools)

        # Initialize EVM agent using SpoonReactAI with EVM tools
        self.evm_agent = SpoonReactAI(
            name="evm_agent",
            description="Intelligent EVM blockchain agent with multi-chain support",
            avaliable_tools=evm_tool_manager
        )

        # Configure supported networks
        self.supported_networks = {
            "ethereum": NetworkConfig(
                name="Ethereum Mainnet",
                chain_id=1,
                rpc_url=os.getenv("ETHEREUM_RPC_URL", "https://eth-mainnet.alchemyapi.io/v2/demo"),
                native_token="ETH",
                explorer_url="https://etherscan.io",
                gas_level="High"
            ),
            "base": NetworkConfig(
                name="Base",
                chain_id=8453,
                rpc_url=os.getenv("BASE_RPC_URL", "https://mainnet.base.org"),
                native_token="ETH",
                explorer_url="https://basescan.org",
                gas_level="Low"
            ),
            "polygon": NetworkConfig(
                name="Polygon",
                chain_id=137,
                rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com/"),
                native_token="MATIC",
                explorer_url="https://polygonscan.com",
                gas_level="Very Low"
            )
        }

        # Default configuration
        self.default_network = "polygon"
        self.confirmation_required = True
        self.debug_mode = False

        # Demo/Test addresses from environment variables
        self.demo_addresses = {
            "primary_wallet": os.getenv("PRIMARY_WALLET", None),
            "secondary_wallet": os.getenv("SECONDARY_WALLET", None),
        }

        # Demo private key from environment variable
        self.private_key = os.getenv("PRIVATE_KEY")

    async def initialize(self):
        """Initialize the agent and EVM tools."""
        logger.info("SpoonOS EVM Command Agent initialized successfully")

    async def process_command(
        self,
        command: str,
        network: Optional[str] = None,
        private_key: Optional[str] = None,
        confirm: Optional[bool] = None,
        debug: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a natural language command and execute the appropriate EVM operation.

        Args:
            command: Natural language command 
            network: Target network name 
            private_key: Private key for transactions (optional, uses env var)
            confirm: Whether to confirm transactions (default: True)
            debug: Enable debug mode (default: False)
            **kwargs: Additional parameters

        Returns:
            Dict containing execution results and metadata
        """
        start_time = datetime.now()

        # Configure parameters
        network = network or self.default_network
        confirm = confirm if confirm is not None else self.confirmation_required
        debug = debug if debug is not None else self.debug_mode

        if debug:
            logger.debug(f"Processing command: '{command}' on network: {network}")

        # Validate network and get config
        if network not in self.supported_networks:
            return {
                "success": False,
                "error": f"Unsupported network '{network}'. Available: {list(self.supported_networks.keys())}",
                "command": command,
                "timestamp": start_time.isoformat()
            }

        config = self.supported_networks[network]
        private_key = private_key or os.getenv("PRIVATE_KEY")

        try:
            # Format the command with context for the EVM agent
            formatted_command = f"""
Execute this EVM blockchain command:
- Command: {command}
- Network: {network} ({config.name})
- Chain ID: {config.chain_id}
- RPC URL: {config.rpc_url}
- Private Key: {'***PROVIDED***' if private_key else 'NOT PROVIDED'}
- Confirm transactions: {confirm}

Please use the appropriate EVM tools to execute this command safely and return the results.
"""

            # Execute the command using EVM agent
            agent_response = await self.evm_agent.run(formatted_command)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Extract transaction hash if present
            tx_hash = None
            explorer_link = None
            if isinstance(agent_response, str):
                import re
                # Look for transaction hash patterns in the response
                hash_pattern = r"0x[a-fA-F0-9]{64}"
                tx_match = re.search(hash_pattern, agent_response)
                if tx_match:
                    tx_hash = tx_match.group(0)
                    explorer_link = f"{config.explorer_url}/tx/{tx_hash}"

            return {
                "success": True,
                "result": {"message": agent_response},
                "command": command,
                "network": config.name,
                "transaction_hash": tx_hash,
                "explorer_link": explorer_link,
                "execution_time_seconds": execution_time,
                "timestamp": start_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}",
                "command": command,
                "network": config.name,
                "execution_time_seconds": execution_time,
                "timestamp": start_time.isoformat()
            }

    async def get_network_status(self, network: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for a specific network or all networks."""
        if network:
            if network not in self.supported_networks:
                return {"error": f"Unknown network: {network}"}

            config = self.supported_networks[network]
            try:
                # Test connectivity
                from web3 import Web3, HTTPProvider
                w3 = Web3(HTTPProvider(config.rpc_url))
                is_connected = w3.is_connected()

                status = {
                    "network": config.name,
                    "chain_id": config.chain_id,
                    "connected": is_connected,
                    "native_token": config.native_token,
                    "gas_level": config.gas_level,
                    "is_testnet": config.is_testnet
                }

                if is_connected:
                    latest_block = w3.eth.block_number
                    status["latest_block"] = latest_block

                return status
            except Exception as e:
                return {
                    "network": config.name,
                    "connected": False,
                    "error": str(e)
                }
        else:
            # Return status for all networks
            all_status = {}
            for net_name in self.supported_networks:
                all_status[net_name] = await self.get_network_status(net_name)
            return all_status

def _print_demo_result(result: Dict[str, Any]):

    is_success = result.get('success', False)

    if is_success:
        result_data = result.get('result', {})
        message = result_data.get('message') if isinstance(result_data, dict) else result_data
        if message and str(message).strip():
            print(f"Result: {message}")
    else:
        error_msg = str(result.get('error', 'Unknown error'))
        print(f"Error: {error_msg[:200]}{'...' if len(error_msg) > 200 else ''}")
    details_to_print = {
        "Transaction": result.get('transaction_hash'),
        "Explorer": result.get('explorer_link'),
        "Network": result.get('network'),
    }
    for label, value in details_to_print.items():
        if value:  
            print(f"{label}: {value}")
    exec_time = result.get('execution_time_seconds')
    if exec_time is not None:
        print(f"Execution time: {exec_time:.2f}s")

async def run_interactive_demo():

    # Initialize the agent
    agent = SpoonEVMCommandAgent()
    await agent.initialize()

    # Comprehensive demo commands to test all EVM toolkit functions
    demo_commands = [
        {
            "command": f"Check ETH balance for {agent.demo_addresses['primary_wallet']}",
            "network": "polygon",
            "description": " EvmBalanceTool - ETH balance query"
        },
        {
            "command": "Get quote for swapping 0.01 ETH to USDC",
            "network": "polygon",
            "description": "💱 EvmSwapQuoteTool - ETH to USDC price quote"
        },
        {
            "command": f"Send 0.0001 ETH to {agent.demo_addresses['primary_wallet']}",
            "network": "polygon",
            "description": " EvmTransferTool - Native ETH transfer "
        },
        {
            "command": f"Send 0.01 USDC to {agent.demo_addresses['primary_wallet']}",
            "network": "polygon",
            "description": " EvmErc20TransferTool - USDC transfer "
        },
        {
            "command": "Swap 0.001 ETH for USDC",
            "network": "polygon",
            "description": " EvmSwapTool - ETH to USDC swap "
        },
        {
            "command": "Bridge 0.01 USDC to Ethereum",
            "network": "polygon",
            "description": " EvmBridgeTool - USDC bridge to Ethereum "
        },
    ]

    for i, demo in enumerate(demo_commands, 1):
        print(f"\n[{i}/{len(demo_commands)}] {demo['description']}")
        print(f"Command: \"{demo['command']}\" (Network: {demo['network']})")
        print("-" * 50)

        try:
            result = await agent.process_command(
                command=demo['command'],
                network=demo['network'],
                private_key=agent.private_key,
                confirm=False,
                debug=False
            )
            _print_demo_result(result)

        except Exception as e:
            print(f" UNHANDLED EXCEPTION: {str(e)[:150]}")

        await asyncio.sleep(2)

async def main():
    """Main entry point - run the demo."""
    await run_interactive_demo()

if __name__ == "__main__":
    # Load environment variables if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    asyncio.run(main())