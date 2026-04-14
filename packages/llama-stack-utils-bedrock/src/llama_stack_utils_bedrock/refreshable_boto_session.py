# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import datetime
from time import time
from uuid import uuid4

from boto3 import Session
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session


class RefreshableBotoSession:
    """
    Boto Helper class which lets us create a refreshable session so that we can cache the client or resource.

    Usage
    -----
    session = RefreshableBotoSession().refreshable_session()

    client = session.client("s3") # we now can cache this client object without worrying about expiring credentials
    """

    def __init__(
        self,
        region_name: str = None,
        profile_name: str = None,
        sts_arn: str = None,
        session_name: str = None,
        session_ttl: int = 30000,
    ):
        """
        Initialize `RefreshableBotoSession`

        Parameters
        ----------
        region_name : str (optional)
            Default region when creating a new connection.

        profile_name : str (optional)
            The name of a profile to use.

        sts_arn : str (optional)
            The role arn to sts before creating a session.

        session_name : str (optional)
            An identifier for the assumed role session. (required when `sts_arn` is given)

        session_ttl : int (optional)
            An integer number to set the TTL for each session. Beyond this session, it will renew the token.
            50 minutes by default which is before the default role expiration of 1 hour
        """

        self.region_name = region_name
        self.profile_name = profile_name
        self.sts_arn = sts_arn
        self.session_name = session_name or uuid4().hex
        self.session_ttl = session_ttl

    def __get_session_credentials(self):
        """
        Get session credentials
        """
        session = Session(region_name=self.region_name, profile_name=self.profile_name)

        # if sts_arn is given, get credential by assuming the given role
        if self.sts_arn:
            sts_client = session.client(service_name="sts", region_name=self.region_name)
            response = sts_client.assume_role(
                RoleArn=self.sts_arn,
                RoleSessionName=self.session_name,
                DurationSeconds=self.session_ttl,
            ).get("Credentials")

            credentials = {
                "access_key": response.get("AccessKeyId"),
                "secret_key": response.get("SecretAccessKey"),
                "token": response.get("SessionToken"),
                "expiry_time": response.get("Expiration").isoformat(),
            }
        else:
            session_credentials = session.get_credentials().get_frozen_credentials()
            credentials = {
                "access_key": session_credentials.access_key,
                "secret_key": session_credentials.secret_key,
                "token": session_credentials.token,
                "expiry_time": datetime.datetime.fromtimestamp(time() + self.session_ttl, datetime.UTC).isoformat(),
            }

        return credentials

    def refreshable_session(self) -> Session:
        """
        Get refreshable boto3 session.
        """
        # Get refreshable credentials
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self.__get_session_credentials(),
            refresh_using=self.__get_session_credentials,
            method="sts-assume-role",
        )

        # attach refreshable credentials current session
        session = get_session()
        session._credentials = refreshable_credentials
        session.set_config_variable("region", self.region_name)
        autorefresh_session = Session(botocore_session=session)

        return autorefresh_session
