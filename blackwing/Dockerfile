FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY *.py case_source_registry.json aviation-glossary.pdf ./
COPY examples ./examples
COPY evaluation ./evaluation
COPY indexes ./indexes
ENV BLACKWING_OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8501
VOLUME ["/app/output"]
CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0", "--server.headless=true"]
