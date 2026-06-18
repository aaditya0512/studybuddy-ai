FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expose port 7860
ENV PORT=7860
EXPOSE 7860

# Run the app with gunicorn
CMD gunicorn -w 2 -b 0.0.0.0:7860 app:app
