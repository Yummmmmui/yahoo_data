FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN chmod +x run_all.sh
CMD ["bash", "run_all.sh"]
