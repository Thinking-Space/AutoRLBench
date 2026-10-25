# autorlbench-image: autorlbench/vl-grpo-cu129-recorded:latest
# Recorded torch2.8/cu129 + vLLM0.11 runtime in an isolated venv.
FROM verlai/verl:vllm017.latest@sha256:4c43bbf17e90284b1102008399240b25406e8d34fea178d86272231b333b7cb6
ENV UV_CACHE_DIR=/opt/build-cache/uv UV_PYTHON_INSTALL_DIR=/opt/uv-python PYTHONPATH="" PYTHONNOUSERSITE=1
RUN python3 -m pip install --no-deps --target /opt/uv-bootstrap uv==0.13.2
RUN /opt/uv-bootstrap/bin/uv venv --python /usr/bin/python3 /opt/recorded-runtime
COPY vl-grpo-recorded-runtime/requirements.txt /opt/recorded-requirements.txt
RUN /opt/uv-bootstrap/bin/uv pip install --python /opt/recorded-runtime/bin/python --no-deps --index https://download.pytorch.org/whl/cu129 -r /opt/recorded-requirements.txt
RUN /opt/uv-bootstrap/bin/uv pip check --python /opt/recorded-runtime/bin/python
RUN /opt/recorded-runtime/bin/python -c 'import sys,torch,vllm,ray,flash_attn,pyarrow; assert sys.version_info[:2]==(3,12); assert torch.__version__=="2.8.0+cu129"; assert torch.version.cuda=="12.9"; assert vllm.__version__=="0.11.0"; assert ray.__version__=="2.49.2"; assert flash_attn.__version__=="2.8.3"; assert pyarrow.__version__=="21.0.0"'
ENV PATH=/opt/recorded-runtime/bin:$PATH
RUN mkdir -p /autorlbench/prepared /autorlbench/models /autorlbench/datasets
RUN printf '%s\n' "$PATH" > /etc/autorlbench-image-path
COPY autorlbench-activate.sh /etc/profile.d/autorlbench-activate.sh
RUN sed -i 's/\r$//' /etc/profile.d/autorlbench-activate.sh && chmod 0644 /etc/profile.d/autorlbench-activate.sh && sh -n /etc/profile.d/autorlbench-activate.sh
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
