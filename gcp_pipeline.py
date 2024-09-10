from kfp import dsl, compiler
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google.oauth2 import service_account
import os

# Set your project details
PROJECT_ID = 'python-code-running'
REGION = 'europe-west1'
BUCKET_NAME = 'main_il'
PIPELINE_ROOT = f'gs://{BUCKET_NAME}/pipeline_root'

# Path to your service account JSON key file
CREDENTIALS_PATH = r"python-code-running-d8cdf4f7fd69.json"  # Replace with the actual path of the key

# Load credentials from the service account JSON key file
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Define the training component
@dsl.component(
    base_image="europe-west1-docker.pkg.dev/python-code-running/ms-mri-train/ms-mri-train:latest"
)
def train_model(data_path: str, output_path: str) -> str:
    import subprocess
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "python3", "gcp_MS_f_2C.py",
        "--data_path", data_path,
        "--output_path", output_path
    ]

    subprocess.run(command, check=True)

    return output_path




# Define the model registration component
@dsl.component(base_image='python:3.9')
def register_model(model_path: str, model_name: str):
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_path,
        serving_container_image_uri=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/ml-models/ms-mri-training:latest"
    )

    print(f"Model registered: {model.resource_name}")


# Define the pipeline
@dsl.pipeline(name="ms-mri-weekly-training", pipeline_root=PIPELINE_ROOT)
def ms_mri_pipeline(
        bucket: str = BUCKET_NAME,
        prefix: str = "MS_MRI_2CLASS/2C - test"
):
    # Call the training component
    train_task = train_model(
        data_path=f"gs://{bucket}/{prefix}",
        output_path=f"gs://{bucket}/models/{{{{PIPELINE_JOB_ID}}}}"
    )

    # Set resource limits for training
    #train_task.set_cpu_limit('32')  # Set CPU limit
    #train_task.set_memory_limit('128Gi')  # Set memory limit
    train_task.add_node_selector_constraint('NVIDIA_TESLA_T4')
    train_task.set_accelerator_limit(1)

    # Register the model
    register_model(
        model_path=train_task.output,
        model_name=f"ms-mri-model-{{{{PIPELINE_JOB_ID}}}}"
    )


# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=ms_mri_pipeline,
    package_path='ms_mri_pipeline.json'
)

# Initialize Vertex AI with credentials
aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)

# Create the pipeline job
job = pipeline_jobs.PipelineJob(
    display_name="ms-mri-weekly-training",
    template_path="ms_mri_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "bucket": BUCKET_NAME,
        "prefix": "MS_MRI_2CLASS/2C - test"
    }
)

# Submit the job
job.submit()

print(f"Pipeline job submitted. Job name: {job.name}")
print(f"Pipeline job details: {job.display_name}")

# Optionally create a schedule for the pipeline
try:
    pipeline_job_schedule = job.create_schedule(
        display_name="Weekly MS MRI Training Schedule",
        cron="0 2 * * 1",  # Run weekly on Monday at 2 AM UTC
        max_concurrent_run_count=1,
    )
    print(f"Schedule created: {pipeline_job_schedule.resource_name}")
except Exception as e:
    print(f"Error creating schedule: {e}")
