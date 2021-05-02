FROM python:3.8.8-slim-buster

RUN apt install libportaudio0 libportaudio2 libportaudiocpp0 portaudio19-dev

COPY requirements.txt /inferenciator/

RUN pip3 install -r /inferenciator/requirements.txt

COPY ./ /inferenciator/

WORKDIR /inferenciator/

CMD ["python3","inferencerStream.py"]
