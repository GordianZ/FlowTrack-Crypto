FROM python:3

WORKDIR /usr/src/app

RUN pip install requests numpy flask

COPY . .

CMD [ "flask", "--app", "./binance_funding_flow_analyzer.py", "run", "--host=0.0.0.0" ]
