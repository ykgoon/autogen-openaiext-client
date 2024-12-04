"""This example demonstrates a human user interacting with a coder agent and a executor agent
to generate and execute code snippets. The user and the agents take turn sequentially
to write input, generate code snippets and execute them, orchestrated by an
round-robin orchestrator agent. The code snippets are executed inside a docker container.
"""

import asyncio
import logging

from autogen_core.application import SingleThreadedAgentRuntime
from autogen_core.application.logging import EVENT_LOGGER_NAME
from autogen_core.base import AgentId, AgentProxy
from autogen_core.components.code_executor import CodeBlock
from autogen_ext.code_executors import DockerCommandLineCodeExecutor
from autogen_magentic_one.agents.coder import Coder, Executor
from autogen_magentic_one.agents.orchestrator import RoundRobinOrchestrator
from autogen_magentic_one.agents.user_proxy import UserProxy
from autogen_magentic_one.messages import RequestReplyMessage
from autogen_magentic_one.utils import LogHandler
from autogen_openaiext_client import OpenAIExtChatCompletionClient
from autogen_openaiext_client.info import GeminiInfo
from dotenv import load_dotenv
import os
import asyncio


async def confirm_code(code: CodeBlock) -> bool:
    response = await asyncio.to_thread(
        input,
        f"Executor is about to execute code (lang: {code.language}):\n{code.code}\n\nDo you want to proceed? (yes/no): ",
    )
    return response.lower() == "yes"


async def main() -> None:
    # Create the runtime.
    runtime = SingleThreadedAgentRuntime()

    load_dotenv()
    assert "GEMINI_API_KEY" in os.environ.keys()

    client = OpenAIExtChatCompletionClient(
        model="gemini-1.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        model_info=GeminiInfo,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    async with DockerCommandLineCodeExecutor() as code_executor:
        # Register agents.
        await Coder.register(runtime, "Coder", lambda: Coder(model_client=client))
        coder = AgentProxy(AgentId("Coder", "default"), runtime)

        await Executor.register(
            runtime,
            "Executor",
            lambda: Executor(
                "A agent for executing code",
                executor=code_executor,
                confirm_execution=confirm_code,
            ),
        )
        executor = AgentProxy(AgentId("Executor", "default"), runtime)

        await UserProxy.register(
            runtime,
            "UserProxy",
            lambda: UserProxy(description="The current user interacting with you."),
        )
        user_proxy = AgentProxy(AgentId("UserProxy", "default"), runtime)

        await RoundRobinOrchestrator.register(
            runtime,
            "orchestrator",
            lambda: RoundRobinOrchestrator([coder, executor, user_proxy]),
        )

        runtime.start()
        await runtime.send_message(RequestReplyMessage(), user_proxy.id)
        await runtime.stop_when_idle()


if __name__ == "__main__":
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    log_handler = LogHandler()
    logger.handlers = [log_handler]
    asyncio.run(main())
