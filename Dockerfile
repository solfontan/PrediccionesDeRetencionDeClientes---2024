FROM python:3.10-slim-bullseye

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir -p app

COPY ./app app2.py

WORKDIR app2

EXPOSE 2424

CMD [ "python3", "app2.py"]