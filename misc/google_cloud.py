from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client('liquid-force-295404')
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=''):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    storage_client = storage.Client('liquid-force-295404')
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)
