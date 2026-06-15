FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY *.py ./
COPY dashboard/ ./dashboard/

ENV PORT=8080
EXPOSE 8080

CMD ["python", "mlb_sbr.py", "dashboard", "--source", "espn", "--no-browser"]
