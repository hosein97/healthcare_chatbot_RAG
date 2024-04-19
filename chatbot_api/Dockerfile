# chatbot_api/Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./src/ /app

EXPOSE 8000
CMD ["sh", "entrypoint.sh"]