# Copyright 2024 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registry of model architectures.

The full registry (with megatron/vision model classes) lives in the complete
verl source distribution; this workspace snapshot only exercises the FSDP +
HuggingFace path, so the registry here is the minimal upstream-compatible
API surface: ``load_model_cls`` and ``get_supported_archs``.
"""

from typing import Optional


class ModelRegistry:
    """Lookup for model classes by ``config.architectures`` entry."""

    @classmethod
    def load_model_cls(cls, arch: str, value: bool = False) -> Optional[type]:
        """Return the registered model class for ``arch``, or None."""
        return None

    @classmethod
    def get_supported_archs(cls) -> list:
        """Return the list of supported architecture names."""
        return []
