FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py config.yml ./

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501" ]