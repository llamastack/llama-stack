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

from llama_stack.apis.version import LLAMA_STACK_API_V1 # noqa: E402
from llama_stack.core.stack import LlamaStack  # noqa: E402

from .pyopenapi.options import Options  # noqa: E402
from .pyopenapi.specification import Info, Server  # noqa: E402
from .pyopenapi.utility import Specification, validate_api  # noqa: E402


def str_presenter(dumper, data):
    if data.startswith(f"/{LLAMA_STACK_API_V1}") or data.startswith(
        "#/components/schemas/"
    ):
        style = None
    else:
        style = ">" if "\n" in data or len(data) > 40 else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


def generate_spec(output_dir: Path, stability_filter: str = None, main_spec: bool = False, combined_spec: bool = False):
    """Generate OpenAPI spec with optional stability filtering."""

    if combined_spec:
        # Special case for combined stable + experimental APIs
        title_suffix = " - Stable & Experimental APIs"
        filename_prefix = "stainless-"
        description_suffix = "\n\n**🔗 COMBINED**: This specification includes both stable production-ready APIs and experimental pre-release APIs. Use stable APIs for production deployments and experimental APIs for testing new features."
        # Use the special "stainless" filter to include stable + experimental APIs
        stability_filter = "stainless"
    elif stability_filter:
        title_suffix = {
            "stable": " - Stable APIs" if not main_spec else "",
            "experimental": " - Experimental APIs",
            "deprecated": " - Deprecated APIs"
        }.get(stability_filter, f" - {stability_filter.title()} APIs")

        # Use main spec filename for stable when main_spec=True
        if main_spec and stability_filter == "stable":
            filename_prefix = ""
        else:
            filename_prefix = f"{stability_filter}-"

        description_suffix = {
            "stable": "\n\n**✅ STABLE**: Production-ready APIs with backward compatibility guarantees.",
            "experimental": "\n\n**🧪 EXPERIMENTAL**: Pre-release APIs (v1alpha, v1beta) that may change before becoming stable.",
            "deprecated": "\n\n**⚠️ DEPRECATED**: Legacy APIs that may be removed in future versions. Use for migration reference only."
        }.get(stability_filter, "")
    else:
        title_suffix = ""
        filename_prefix = ""
        description_suffix = ""

    spec = Specification(
        LlamaStack,
        Options(
            server=Server(url="http://any-hosted-llama-stack.com"),
            info=Info(
                title=f"Llama Stack Specification{title_suffix}",
                version=LLAMA_STACK_API_V1,
                description=f"""This is the specification of the Llama Stack that provides
                a set of endpoints and their corresponding interfaces that are tailored to
                best leverage Llama Models.{description_suffix}""",
            ),
            include_standard_error_responses=True,
            stability_filter=stability_filter,  # Pass the filter to the generator
        ),
    )

    yaml_filename = f"{filename_prefix}llama-stack-spec.yaml"

    with open(output_dir / yaml_filename, "w", encoding="utf-8") as fp:
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
    print(f"Converting the spec to YAML (openapi.yaml) and HTML (openapi.html) at {now}")
    print("")

    # Generate main spec as stable APIs (llama-stack-spec.yaml)
    print("Generating main specification (stable APIs)...")
    generate_spec(output_dir, "stable", main_spec=True)

    print("Generating other stability-filtered specifications...")
    generate_spec(output_dir, "experimental")
    generate_spec(output_dir, "deprecated")

    print("Generating combined stable + experimental specification...")
    generate_spec(output_dir, combined_spec=True)


if __name__ == "__main__":
    fire.Fire(main)
