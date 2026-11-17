#!/usr/bin/env bash
# cctl wrapper: strip proxy env vars and force server endpoint.
# Note: 'job stop' in this version requires interactive confirmation.
# Workaround: add --yes flag to skip confirmation.
exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  CCTL_SERVER=https://cybertron.modelbest.co /user/xuxiaoang/.local/bin/cctl "$@"
