# Logs always land in outputs/rerun/<task_id>/<timestamp>/ (run.log, prepare.log,
# rerun.log, command.sh, report.json, plus the experiment's results/ and the workspace).
python3 -m autorlbench rerun tasks/algorithm/verl_grpo_advantage_broadcast \
  --huggingface-mirror https://hf-mirror.com --log-level debug


AUTH=$(printf '100054916977:Genshinyha180!!!' | base64)
jq --arg a "$AUTH" '.auths["ccr.ccs.tencentyun.com"]={auth:$a}' \
  ~/.docker/config.json > /tmp/dc.json && mv /tmp/dc.json ~/.docker/config.json
