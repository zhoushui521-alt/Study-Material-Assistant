"""知行 Stage 5.5 Model Gateway 公共入口。"""

from app.model_gateway.errors import (
    ModelAuthenticationError,
    ModelCredentialDecryptionError,
    ModelCredentialEncryptionUnavailableError,
    ModelCredentialError,
    ModelCredentialStoreError,
    ModelGatewayConfigurationError,
    ModelGatewayError,
    UnsupportedModelProviderError,
)
from app.model_gateway.gateway import (
    ModelGateway,
    ModelSelection,
    is_model_authentication_error,
    normalize_api_key,
    normalize_model_name,
    normalize_provider,
)
from app.model_gateway.providers import (
    ModelRoute,
    OpenAICompatibleProviderAdapter,
    ProviderAdapter,
    ProviderAdapterRegistry,
)

__all__ = [
    "ModelAuthenticationError",
    "ModelCredentialDecryptionError",
    "ModelCredentialEncryptionUnavailableError",
    "ModelCredentialError",
    "ModelCredentialStoreError",
    "ModelGateway",
    "ModelGatewayConfigurationError",
    "ModelGatewayError",
    "ModelRoute",
    "ModelSelection",
    "OpenAICompatibleProviderAdapter",
    "ProviderAdapter",
    "ProviderAdapterRegistry",
    "UnsupportedModelProviderError",
    "is_model_authentication_error",
    "normalize_api_key",
    "normalize_model_name",
    "normalize_provider",
]
