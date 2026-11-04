# 算力排队脚本使用说明

用于阶段 3。已有团队调度队列时优先使用其分配机制；直接等待服务器空闲 GPU 时使用以下脚本。本机运行脚本，通过 SSH 查询远程实验服务器；算力就绪后唤醒本机当前聊天。

以下命令在本机 PowerShell 中执行，参数按实验方案填写。

## 1. 创建队列

```powershell
$python = 'C:\path\to\python.exe'
$skill = 'C:\path\to\task-construction'
$caseDir = 'C:\path\to\workspace\cases\example'
$queueDir = Join-Path $caseDir 'queue'
$state = Join-Path $queueDir 'state.json'
$sshTarget = 'gpu-server'

& $python "$skill\scripts\gpu_queue.py" init `
  --state $state `
  --ssh-target $sshTarget `
  --gpu-count 2 `
  --gpu-name A100 `
  --min-free-mib 40000
```

| 参数 | 用法 |
| --- | --- |
| `--state` | 本机排队状态文件路径，各命令使用同一路径 |
| `--ssh-target` | 远程服务器的 SSH 别名或 `user@host` |
| `--gpu-count` | 所需 GPU 数量 |
| `--gpu-name` | GPU 型号筛选，默认不限 |
| `--gpu-ids` | 允许的 GPU 索引或 UUID，以逗号分隔，默认不限 |
| `--min-free-mib` | 每卡最低空闲显存，默认 `0` MiB |
| `--max-used-mib` | 每卡显存占用上限，默认 `5120` MiB（5 GiB） |
| `--max-utilization` | GPU 利用率上限，默认 `10`% |
| `--interval-minutes` | 检查间隔，默认 `15` 分钟 |
| `--nvidia-smi` | 远程可执行路径，默认 `nvidia-smi` |

等待时长不限。单次检查满足配置要求且 GPU 无计算进程时，状态即为 `ready`。

## 2. 后台等待并唤醒

沿用上述变量，将 `$thread` 替换为当前聊天 UUID：

```powershell
$thread = '<current-chat-uuid>'
$watcherArgs = @(
  '-u', ('"{0}"' -f "$skill\scripts\watch_gpu_queue.py"),
  '--state', ('"{0}"' -f $state),
  '--thread', $thread,
  '--cwd', ('"{0}"' -f $caseDir)
)
$watcher = Start-Process -FilePath $python -ArgumentList $watcherArgs `
  -WorkingDirectory $caseDir -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput "$queueDir\watcher.out.log" `
  -RedirectStandardError "$queueDir\watcher.err.log"
$watcher.Id
```

`--cwd` 为本机 case 目录。需要指定 Codex CLI 入口时，添加 `--codex-command-json`，值为命令数组，例如 `["node", "C:/path/to/codex.js"]`。

查看 `queue/` 下的日志和 `state.json` 获取等待状态，记录 watcher PID。等待期间使用后台排队脚本；运行监督 heartbeat 在实验启动后创建。就绪后 watcher 向指定聊天提交继续执行消息并退出；投递失败时查看错误日志，保留队列状态并重新启动 watcher。

## 3. 就绪后启动实验

```powershell
& $python "$skill\scripts\gpu_queue.py" claim --state $state
```

唤醒后核对当前 case 的作业记录，已有运行时继续监督。本次输出为 `status=claimed` 且 `probed=true` 时启动远程实验，实验启动后清理本次等待进程。

## 4. 手动检查与取消

```powershell
& $python "$skill\scripts\gpu_queue.py" check --state $state
& $python "$skill\scripts\gpu_queue.py" cancel --state $state
```

`check` 立即检查；`cancel` 取消等待并使 watcher 退出。

状态：`waiting` 等待、`ready` 就绪、`claimed` 算力已确认、`cancelled` 已取消、`error` 检查失败。命令退出码：`0` 正常，`2` 错误。
