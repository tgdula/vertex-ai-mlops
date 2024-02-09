# vertex-ai-mlops

Using Vertex AI for Machine Learning

## resources

* official documentation: https://cloud.google.com/vertex-ai
  * [blob storage reference](https://cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob)
  * [exporting models for prediction](https://cloud.google.com/ai-platform/prediction/docs/exporting-for-prediction#joblib)
    > HINT: model artifact's filename must exactly match one of specified options!
  * [troubleshoot container image issues](https://cloud.google.com/artifact-registry/docs/docker/troubleshoot)
* official Google Cloud Platform Vertex AI samples: https://github.com/GoogleCloudPlatform/vertex-ai-samples
* tensorflow blog: [blog.tensorflow.org](https://blog.tensorflow.org)
  * [5 steps to go from a notebook to a deployed model](https://blog.tensorflow.org/2022/05/5-steps-to-go-from-notebook-to-deployed.html): end to end example of model deployment (custom tensorflow model of multiclass classification of flower images)

## useful terminal commands

### gsutil
show all the models:
```
gsutil ls gs://my-bucket/**model**
```


### `jupyter nbconvert` - convert Jupyter notebooks to python: from experiment to job
```
jupyter nbconvert my-notebook.ipynb --to python
```