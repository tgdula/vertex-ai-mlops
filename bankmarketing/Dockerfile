FROM gcr.io/deeplearning-platform-release/rapids-gpu

WORKDIR /

COPY trainer /trainer

ENTRYPOINT ["python", "-m", "trainer.task"]