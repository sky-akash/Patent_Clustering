# Use official Python runtime as a base image
FROM python:3.11.9
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    wget \
    git \
    && apt-get clean

# Install NLTK dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Copy your code
COPY . .

# Expose the Streamlit port
EXPOSE 8080

# Run the Streamlit app
CMD ["streamlit", "run", "patent_class_app.py", "--server.port=8080", "--server.headless=true"]