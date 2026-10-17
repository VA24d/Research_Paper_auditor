# The math sandbox: runs code written by the model, never repository code (decision D11).
# Built and run by auditor.infra.sandbox with no network, a read-only file system and
# CPU, memory, process and time limits.
FROM python:3.13-slim
RUN pip install --no-cache-dir sympy==1.13.3 numpy==2.2.6 mpmath==1.3.0 networkx==3.4.2 z3-solver==4.15.1.0
USER 65534:65534
WORKDIR /tmp
