FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# set path to our python api file
ENV MODULE_NAME="app.main"

# copy contents of project into docker
COPY ./ /app
COPY ./models /models

# install poetry
RUN pip install poetry

# disable virtualenv for peotry
RUN poetry config virtualenvs.create false

# install dependencies
RUN poetry install