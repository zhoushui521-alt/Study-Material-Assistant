"""Model Gateway 对业务层暴露的稳定错误类型。"""

from __future__ import annotations


class ModelGatewayError(RuntimeError):
    """模型路由、凭证或 Provider Adapter 无法完成请求。"""


class ModelGatewayConfigurationError(ModelGatewayError):
    """服务端模型或加密配置无效。"""


class ModelCredentialError(ModelGatewayError):
    """用户模型凭证无法安全保存或读取。"""


class ModelCredentialEncryptionUnavailableError(ModelCredentialError):
    """服务端没有可用的稳定加密主密钥。"""


class ModelCredentialDecryptionError(ModelCredentialError):
    """已保存的密文无法由当前主密钥解密。"""


class ModelCredentialStoreError(ModelCredentialError):
    """模型凭证数据库不可用。"""


class UnsupportedModelProviderError(ModelGatewayError):
    """请求了未注册或未安全配置的 Provider。"""


class ModelAuthenticationError(ModelGatewayError):
    """Provider 拒绝模型凭证；不携带上游错误详情。"""
