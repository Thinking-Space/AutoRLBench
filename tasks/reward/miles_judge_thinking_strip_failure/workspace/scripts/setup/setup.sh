# SKIP_PIP_INSTALL=1 : skip all pip installs (use when docker already has deps)
if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  rm -f /etc/xdg/pip/pip.conf /etc/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf 2>/dev/null || true
  rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED 2>/dev/null || true
  rm -f /etc/pip/constraint.txt 2>/dev/null || true
  touch /etc/pip/constraint.txt

  if [ "${CLUSTER:-}" == "ningxia_h100" ]; then
     pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  elif [ "${CLUSTER:-}" == "kunming_train" ]; then
     pip config set global.index-url https://nexus.xn-01.zetyun.cn/repository/pypi/simple
  else
     pip config set global.index-url https://pypi.hs1.paratera.com/root/pypi/+simple
  fi
  pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  if [ -f /user/panyinxu/tools/block_sparse_attn-0.0.1-cp312-cp312-linux_x86_64.whl ]; then
    pip install /user/panyinxu/tools/block_sparse_attn-0.0.1-cp312-cp312-linux_x86_64.whl
  else
    echo "[setup.sh] skip missing wheel: /user/panyinxu/tools/block_sparse_attn-0.0.1-cp312-cp312-linux_x86_64.whl"
  fi
  if [ -f /user/panyinxu/tools/nsa-0.0.0-cp312-cp312-linux_x86_64.whl ]; then
    pip install /user/panyinxu/tools/nsa-0.0.0-cp312-cp312-linux_x86_64.whl
  else
    echo "[setup.sh] skip missing wheel: /user/panyinxu/tools/nsa-0.0.0-cp312-cp312-linux_x86_64.whl"
  fi
  # 检查 sgl-kernel 或 sglang-kernel 的版本，优先 sgl-kernel，再查 sglang-kernel
  cur_sgl_ver=$(pip show sgl-kernel 2>/dev/null | awk '/^Version:/{print $2}')
  if [ -z "$cur_sgl_ver" ]; then
    cur_sgl_ver=$(pip show sglang-kernel 2>/dev/null | awk '/^Version:/{print $2}')
  fi
  if [ -z "$cur_sgl_ver" ] || python3 -c "from packaging.version import Version; exit(0 if Version('$cur_sgl_ver') < Version('0.3.20') else 1)" 2>/dev/null; then
    pip install -U "sgl-kernel>=0.3.20" || echo "[setup.sh] sgl-kernel install failed, continuing..."
  else
    echo "[setup.sh] sgl-kernel $cur_sgl_ver >= 0.3.20, skip upgrade."
  fi
  pip install timeout_decorator polars seaborn
  pip install google-cloud-aiplatform latex2sympy2 pylatexenc tabulate vertexai
  pip install math-verify antlr4-python3-runtime -U
  if [ -f /user/panyinxu/tools/omegaconf-2.4.0.dev3-py3-none-any.whl ]; then
    pip install /user/panyinxu/tools/omegaconf-2.4.0.dev3-py3-none-any.whl
  else
    echo "[setup.sh] skip missing wheel: /user/panyinxu/tools/omegaconf-2.4.0.dev3-py3-none-any.whl"
  fi
  if [ -f /user/panyinxu/tools/hydra_core-1.4.0.dev1-py3-none-any.whl ]; then
    pip install /user/panyinxu/tools/hydra_core-1.4.0.dev1-py3-none-any.whl
  else
    echo "[setup.sh] skip missing wheel: /user/panyinxu/tools/hydra_core-1.4.0.dev1-py3-none-any.whl"
  fi
  pip install grpcio grpcio-tools protobuf loguru
  pip install swanlab
  echo "[setup.sh] pip installs completed."
else
  echo "[setup.sh] SKIP_PIP_INSTALL=1, skipping all pip installs."
fi

cd Megatron-LM
OLD_MARKER=".miles_megatron_patch_v0.5.7_applied"
PATCH_MARKER=".miles_megatron_patch_latest_applied"
if [ -f "${OLD_MARKER}" ] && [ ! -f "${PATCH_MARKER}" ]; then
  mv "${OLD_MARKER}" "${PATCH_MARKER}"
  echo "migrated patch marker: ${OLD_MARKER} -> ${PATCH_MARKER}"
fi
if [ -f "${PATCH_MARKER}" ]; then
  echo "skip megatron.patch (marker exists: ${PATCH_MARKER})"
else
  patch -p1 -N --batch < "$WORK_DIR/docker/patch/latest/megatron.patch" || true
  touch "${PATCH_MARKER}"
  echo "applied megatron.patch (created marker: ${PATCH_MARKER})"
fi
