FROM bitnami/pytorch:latest

RUN apt-get install build-essential
RUN apt-get install python-dev
RUN apt-get install python-setuptools
COPY requirements.txt /

RUN python3 -m pip install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000
CMD [ "python" , "app.py"]
