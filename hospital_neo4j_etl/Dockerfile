FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

CMD ["sh", "entrypoint.sh"]