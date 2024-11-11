import os
import boto3
from fastapi import UploadFile

async def init_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET"),
        endpoint_url=os.getenv("AWS_URL"),
        region_name=os.getenv("AWS_REGION")
    )


async def store_to_digital_ocean(
        file: UploadFile,
):
    """
        Stores a file to Digital Ocean Spaces
    """
    try:
        s3_client = await init_s3_client()
        BUCKET_NAME = os.getenv("AWS_BUCKET")


        # File upload to D.O spaces
        s3_client.upload_fileobj(
            file.file,
            BUCKET_NAME,
            file.filename,
            ExtraArgs={"ACL": "public-read", "ContentType": file.content_type}
        )

        return f"{os.getenv('AWS_URL')}/{BUCKET_NAME}/{file.filename}"
    except Exception as exc:
        raise Exception(str(exc))