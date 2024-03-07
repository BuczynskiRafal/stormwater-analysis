FROM python:3.10.13-slim-bullseye

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV DJANGO_SETTINGS_MODULE=storm_analysis.settings

RUN python manage.py migrate 

EXPOSE 8000

CMD [python manage.py runserver 0.0.0.0:8000]