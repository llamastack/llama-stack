# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Base class for mock servers with async startup support.

All mock servers should inherit from MockServerBase and implement await_start().
"""

from abc import abstractmethod

from pydantic import BaseModel


class MockServerBase(BaseModel):
    """
    Pydantic base model for mock servers.

    **TO CREATE A NEW MOCK SERVER:**
    1. Inherit from this class
    2. Implement async def await_start(self)
    3. Implement def stop(self)
    4. Done!

    Example:
        class MyMockServer(MockServerBase):
            port: int = 8080

            async def await_start(self):
                # Start your server
                self.server = create_server()
                self.server.start()
                # Wait until ready (can check internal state, no HTTP needed)
                while not self.server.is_listening():
                    await asyncio.sleep(0.1)

            def stop(self):
                if self.server:
                    self.server.stop()
    """

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    async def await_start(self):
        """
        Start the server and wait until it's ready.

        This method should:
        1. Start the server (synchronous or async)
        2. Wait until the server is fully ready to accept requests
        3. Return when ready

        Subclasses can check internal state directly - no HTTP polling needed!
        """
        ...

    @abstractmethod
    def stop(self):
        """
        Stop the server and clean up resources.

        This method should gracefully shut down the server.
        """
        ...
