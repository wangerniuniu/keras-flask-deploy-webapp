FROM python:3.7.15-alpine3.17

COPY requirements.txt /
RUN python3 -m pip install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD [ "python" , "app.py"]
