FROM python:3

WORKDIR /usr/src/app

RUN pip install requests pandas numpy streamlit 

COPY . .

CMD [ "python", "./your-daemon-or-script.py" ]
