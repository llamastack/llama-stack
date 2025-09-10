# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
from pathlib import Path
import sys
import fire
import ruamel.yaml as yaml

from llama_stack.apis.version import LLAMA_STACK_API_VERSION  # noqa: E402
from llama_stack.core.stack import LlamaStack  # noqa: E402

from .pyopenapi.options import Options  # noqa: E402
from .pyopenapi.specification import Info, Server  # noqa: E402
from .pyopenapi.utility import Specification, validate_api  # noqa: E402


def str_presenter(dumper, data):
    if data.startswith(f"/{LLAMA_STACK_API_VERSION}") or data.startswith(
        "#/components/schemas/"
    ):
        style = None
    else:
        style = ">" if "\n" in data or len(data) > 40 else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


def main(output_dir: str):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise ValueError(f"Directory {output_dir} does not exist")

    # Validate API protocols before generating spec
    return_type_errors = validate_api()
    if return_type_errors:
        print("\nAPI Method Return Type Validation Errors:\n")
        for error in return_type_errors:
            print(error, file=sys.stderr)
        sys.exit(1)
    now = str(datetime.now())
    print(
        "Converting the spec to YAML (openapi.yaml) and HTML (openapi.html) at " + now
    )
    print("")

    spec = Specification(
        LlamaStack,
        Options(
            server=Server(url="http://any-hosted-llama-stack.com"),
            info=Info(
                title="Llama Stack Specification",
                version=LLAMA_STACK_API_VERSION,
                description="""This is the specification of the Llama Stack that provides
                a set of endpoints and their corresponding interfaces that are tailored to
                best leverage Llama Models.""",
            ),
            include_standard_error_responses=True,
        ),
    )

    with open(output_dir / "llama-stack-spec.yaml", "w", encoding="utf-8") as fp:
        y = yaml.YAML()
        y.default_flow_style = False
        y.block_seq_indent = 2
        y.map_indent = 2
        y.sequence_indent = 4
        y.sequence_dash_offset = 2
        y.width = 80
        y.allow_unicode = True
        y.representer.add_representer(str, str_presenter)

        y.dump(
            spec.get_json(),
            fp,
        )

    with open(output_dir / "llama-stack-spec.html", "w") as fp:
        spec.write_html(fp, pretty_print=True)


if __name__ == "__main__":
    fire.Fire(main)
