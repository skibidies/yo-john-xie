# Mech Royale bot. Build: docker build -t my-bot .
# The match runs it with no network, a read-only disk (writable /tmp), 1 CPU, 512 MB, uid 65534, map at /arena.
FROM python:3.13.1-slim
# numpy is installed system-wide (not --user) so the unprivileged match user can import it.
RUN pip install --no-cache-dir numpy==2.2.6
WORKDIR /bot
COPY mechfight.py worldmap.py bot.py /bot/
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
