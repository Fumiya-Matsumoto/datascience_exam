FROM python:3.8.6-slim-buster

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y libgomp1

COPY ./app /app

WORKDIR /app

RUN pip3 install --upgrade pip \
    && pip3 install -r /app/requirements.txt --no-cache-dir

ENTRYPOINT ["python3"]

CMD ["app.py"]