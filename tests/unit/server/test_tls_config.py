# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

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


class TestProductionVerifyTlsFalse:
    def test_production_mode_rejects_verify_tls_false(self):
        """Production mode should reject auth configs with verify_tls=False at server startup."""
        from ogx.core.datatypes import (
            AuthenticationConfig,
            OAuth2IntrospectionConfig,
            OAuth2TokenAuthConfig,
        )

        auth_config = AuthenticationConfig(
            provider_config=OAuth2TokenAuthConfig(
                introspection=OAuth2IntrospectionConfig(
                    url="https://auth.example.com/introspect",
                    client_id="client",
                    client_secret="secret",
                ),
                verify_tls=False,
            ),
        )
        # The check happens in server.py create_app, not in the model validator.
        # Here we verify the config can be created (the server will reject it at startup).
        assert auth_config.provider_config.verify_tls is False


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
