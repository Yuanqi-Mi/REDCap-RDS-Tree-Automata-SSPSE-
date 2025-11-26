FROM rocker/r-ver:4.3.1
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN Rscript -e "install.packages('RDS', repos='https://cloud.r-project.org')"
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt
COPY . /app/
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
