# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.datatypes import (
    FIPS_APPROVED_CIPHERS,
    SecurityMode,
    ServerConfig,
    ServerTLSConfig,
    StackConfig,
)


class TestSecurityModeValidation:
    def test_development_mode_no_certs_allowed(self):
        """Development mode should work without TLS certificates."""
        config = ServerConfig(security_mode=SecurityMode.DEVELOPMENT)
        assert config.security_mode == SecurityMode.DEVELOPMENT
        assert config.tls_certfile is None
        assert config.tls_keyfile is None

    def test_production_mode_requires_certs(self):
        """Production mode should raise ValueError without TLS certificates."""
        with pytest.raises(ValueError, match="Production security mode requires TLS"):
            ServerConfig(security_mode=SecurityMode.PRODUCTION)

    def test_production_mode_requires_both_certs(self):
        """Production mode should require both cert and key."""
        with pytest.raises(ValueError, match="Production security mode requires TLS"):
            ServerConfig(
                security_mode=SecurityMode.PRODUCTION,
                tls_certfile="/path/to/cert.pem",
            )

    def test_production_mode_with_certs(self):
        """Production mode should succeed with both cert and key."""
        config = ServerConfig(
            security_mode=SecurityMode.PRODUCTION,
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
        )
        assert config.security_mode == SecurityMode.PRODUCTION

    def test_production_mode_auto_populates_fips_ciphers(self):
        """Production mode should auto-populate FIPS-approved cipher suites."""
        config = ServerConfig(
            security_mode=SecurityMode.PRODUCTION,
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
        )
        assert config.tls_config is not None
        assert config.tls_config.ciphers == FIPS_APPROVED_CIPHERS

    def test_production_mode_auto_populates_ciphers_when_tls_config_has_none(self):
        """Production mode should fill in ciphers when tls_config exists but ciphers is None."""
        config = ServerConfig(
            security_mode=SecurityMode.PRODUCTION,
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(),
        )
        assert config.tls_config.ciphers == FIPS_APPROVED_CIPHERS

    def test_production_mode_preserves_custom_ciphers(self):
        """Production mode should not overwrite explicitly set ciphers."""
        custom_ciphers = ["ECDHE-RSA-AES256-GCM-SHA384"]
        config = ServerConfig(
            security_mode=SecurityMode.PRODUCTION,
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(ciphers=custom_ciphers),
        )
        assert config.tls_config.ciphers == custom_ciphers

    def test_development_mode_does_not_auto_populate_tls_config(self):
        """Development mode should not auto-create tls_config."""
        config = ServerConfig(security_mode=SecurityMode.DEVELOPMENT)
        assert config.tls_config is None

    def test_default_security_mode_is_development(self):
        """Default security mode should be development."""
        config = ServerConfig()
        assert config.security_mode == SecurityMode.DEVELOPMENT

    def test_production_mode_rejects_non_fips_ciphers(self):
        """Production mode should reject cipher suites not in the FIPS-approved list."""
        with pytest.raises(ValueError, match="FIPS-approved ciphers"):
            ServerConfig(
                security_mode=SecurityMode.PRODUCTION,
                tls_certfile="/path/to/cert.pem",
                tls_keyfile="/path/to/key.pem",
                tls_config=ServerTLSConfig(ciphers=["RC4-SHA"]),
            )

    def test_production_mode_rejects_mixed_ciphers(self):
        """Production mode should reject a list containing any non-FIPS cipher."""
        with pytest.raises(ValueError, match="RC4-SHA"):
            ServerConfig(
                security_mode=SecurityMode.PRODUCTION,
                tls_certfile="/path/to/cert.pem",
                tls_keyfile="/path/to/key.pem",
                tls_config=ServerTLSConfig(ciphers=["ECDHE-RSA-AES256-GCM-SHA384", "RC4-SHA"]),
            )

    def test_production_mode_allows_fips_subset(self):
        """Production mode should accept a subset of FIPS-approved ciphers."""
        subset = ["ECDHE-RSA-AES128-GCM-SHA256", "ECDHE-RSA-AES256-GCM-SHA384"]
        config = ServerConfig(
            security_mode=SecurityMode.PRODUCTION,
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(ciphers=subset),
        )
        assert config.tls_config.ciphers == subset

    def test_production_mode_rejects_empty_ciphers(self):
        """Production mode should reject an empty cipher list."""
        with pytest.raises(ValueError, match="At least one cipher suite"):
            ServerConfig(
                security_mode=SecurityMode.PRODUCTION,
                tls_certfile="/path/to/cert.pem",
                tls_keyfile="/path/to/key.pem",
                tls_config=ServerTLSConfig(ciphers=[]),
            )

    def test_development_mode_allows_any_ciphers(self):
        """Development mode should not enforce FIPS cipher restrictions."""
        config = ServerConfig(
            security_mode=SecurityMode.DEVELOPMENT,
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(ciphers=["RC4-SHA"]),
        )
        assert config.tls_config.ciphers == ["RC4-SHA"]


class TestInsecureFlagOverride:
    def test_insecure_overrides_production_in_raw_config(self):
        """--insecure should override production mode in YAML before validation."""
        raw_config = {
            "version": 2,
            "distro_name": "test",
            "providers": {},
            "server": {"security_mode": "production"},
        }
        # Simulate CLI override applied to raw dict before StackConfig construction
        raw_config["server"]["security_mode"] = "development"

        config = StackConfig(**raw_config)
        assert config.server.security_mode == SecurityMode.DEVELOPMENT

    def test_production_mode_fails_without_insecure(self):
        """Without --insecure, production mode without certs should fail at construction."""
        raw_config = {
            "version": 2,
            "distro_name": "test",
            "providers": {},
            "server": {"security_mode": "production"},
        }
        with pytest.raises(ValueError, match="Production security mode requires TLS"):
            StackConfig(**raw_config)

    def test_env_var_override_rescues_production_config(self):
        """OGX_SECURITY_MODE env var should override production mode before validation."""
        raw_config = {
            "version": 2,
            "distro_name": "test",
            "providers": {},
            "server": {"security_mode": "production"},
        }
        # Simulate what create_app() does: apply env var override before StackConfig construction
        raw_config["server"]["security_mode"] = "development"
        config = StackConfig(**raw_config)
        assert config.server.security_mode == SecurityMode.DEVELOPMENT


class TestProductionVerifyTlsFalse:
    def _make_config_with_verify_tls(self, security_mode, verify_tls):
        """Build a StackConfig with auth provider verify_tls setting."""
        from ogx.core.datatypes import (
            AuthenticationConfig,
            OAuth2IntrospectionConfig,
            OAuth2TokenAuthConfig,
        )

        server_kwargs = {"security_mode": security_mode}
        if security_mode == "production":
            server_kwargs["tls_certfile"] = "/path/to/cert.pem"
            server_kwargs["tls_keyfile"] = "/path/to/key.pem"

        server_kwargs["auth"] = AuthenticationConfig(
            provider_config=OAuth2TokenAuthConfig(
                introspection=OAuth2IntrospectionConfig(
                    url="https://auth.example.com/introspect",
                    client_id="client",
                    client_secret="secret",
                ),
                verify_tls=verify_tls,
            ),
        )
        return StackConfig(
            version=2,
            distro_name="test",
            providers={},
            server=server_kwargs,
        )

    def test_production_mode_rejects_verify_tls_false(self):
        """Production mode should raise SystemExit when verify_tls=False."""
        from ogx.core.server.server import validate_auth_security

        config = self._make_config_with_verify_tls("production", verify_tls=False)
        with pytest.raises(SystemExit, match="verify_tls=False"):
            validate_auth_security(config)

    def test_development_mode_warns_verify_tls_false(self):
        """Development mode should warn but not crash when verify_tls=False."""
        from ogx.core.server.server import validate_auth_security

        config = self._make_config_with_verify_tls("development", verify_tls=False)
        with patch("ogx.core.server.server.logger") as mock_logger:
            validate_auth_security(config)
            mock_logger.warning.assert_called_once()

    def test_verify_tls_true_passes_in_production(self):
        """Production mode with verify_tls=True should pass without error."""
        from ogx.core.server.server import validate_auth_security

        config = self._make_config_with_verify_tls("production", verify_tls=True)
        validate_auth_security(config)


class TestHSTSMiddleware:
    def test_hsts_header_default_max_age(self):
        """HSTS header should use default max-age of 1 year."""
        from ogx.core.server.server import HSTSMiddleware

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(HSTSMiddleware)
        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 200
        assert response.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"

    def test_hsts_header_custom_max_age(self):
        """HSTS header should respect custom max-age."""
        from ogx.core.server.server import HSTSMiddleware

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(HSTSMiddleware, max_age=86400)
        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 200
        assert response.headers["strict-transport-security"] == "max-age=86400; includeSubDomains"

    def test_hsts_header_absent_without_middleware(self):
        """HSTS header should not be present when middleware is not added."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 200
        assert "strict-transport-security" not in response.headers

    def test_hsts_max_age_config_default(self):
        """ServerConfig hsts_max_age should default to 1 year."""
        config = ServerConfig()
        assert config.hsts_max_age == 31536000

    def test_hsts_max_age_config_zero_disables(self):
        """hsts_max_age=0 should be valid (used to disable HSTS)."""
        config = ServerConfig(hsts_max_age=0)
        assert config.hsts_max_age == 0

    def test_hsts_max_age_config_rejects_negative(self):
        """hsts_max_age must not be negative."""
        with pytest.raises(ValueError):
            ServerConfig(hsts_max_age=-1)


class TestFIPSCipherConstants:
    def test_fips_ciphers_are_aes_gcm_only(self):
        """All FIPS-approved ciphers should be AES-GCM variants."""
        for cipher in FIPS_APPROVED_CIPHERS:
            assert "GCM" in cipher, f"Cipher {cipher} is not AES-GCM"

    def test_fips_ciphers_no_chacha(self):
        """CHACHA20-POLY1305 should not be in FIPS-approved list."""
        for cipher in FIPS_APPROVED_CIPHERS:
            assert "CHACHA" not in cipher, f"Cipher {cipher} should not be in FIPS list"

    def test_server_tls_config_defaults(self):
        """ServerTLSConfig should have sensible defaults."""
        config = ServerTLSConfig()
        assert config.ciphers is None
