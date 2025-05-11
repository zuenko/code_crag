# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Set the working directory in the container
WORKDIR ${APP_HOME}

# Install system dependencies if any (e.g. for faiss if not using pre-built wheels)
# RUN apt-get update && apt-get install -y --no-install-recommends some-dependency && rm -rf /var/lib/apt/lists/*

# Install pipenv (or poetry, or just use requirements.txt)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app ${APP_HOME}/app
COPY .env.example ${APP_HOME}/.env.example
# It's better to mount the actual .env file via docker-compose for secrets

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application
# Use uvicorn to run the Litestar application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]