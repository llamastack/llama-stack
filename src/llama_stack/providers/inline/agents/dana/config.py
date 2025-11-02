from typing import Any

from pydantic import BaseModel


class DanaAgentConfig(BaseModel):
    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {}
