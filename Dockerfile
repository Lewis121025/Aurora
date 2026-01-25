FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -e ".[api]"
EXPOSE 8000
CMD ["uvicorn", "aurora.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
