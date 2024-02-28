FROM airis_base

WORKDIR /app

# Copy your Python script into the container at /app
COPY ./main.py /app/
COPY ./baseline_0 /app/baseline_0
COPY ./baseline_1 /app/baseline_1
COPY ./baseline_2 /app/baseline_2
# Install python requirements
# RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "main.py"]

