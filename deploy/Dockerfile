# The hosted dashboard. `scripts/publishSpace.sh` copies this file to the root of the
# staged upload, so the build context is that directory and every path here is relative
# to it.
#
# The image holds no market data and no weights of its own. Both arrive in the upload:
# one results directory and one `trainedPolicies/` tree.
FROM python:3.11-slim

# Hugging Face Spaces runs the container as uid 1000. The forecast screen writes a
# parquet cache under `data_cache/`, so the application directory has to belong to that
# user. HOME has to be writable as well, because matplotlib and torch both write there.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PORT=7860

RUN useradd --create-home --uid 1000 user

WORKDIR /app

# Dependencies first, so a change to the code or the data does not rebuild this layer.
COPY deploy/requirements.txt ./deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt

COPY . .
RUN chown -R user:user /app

USER user

EXPOSE 7860

# One worker: the app caches a region's rows in process memory, and a second copy of
# torch buys nothing on two cores. Threads carry the concurrent requests. The timeout
# is long because the forecast screen fetches from NEMWeb on the request.
CMD ["gunicorn", "--chdir", "/app", "--pythonpath", "/app", \
     "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", \
     "--timeout", "180", "--access-logfile", "-", "deploy.app:server"]
