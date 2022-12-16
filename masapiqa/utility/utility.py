import os
import zipfile
from pathlib import Path
import requests
from urllib.parse import urlparse
from tqdm import tqdm


def download(url, file_path):
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)

    scheme = urlparse(url).scheme

    if scheme == "":
        url = "http://" + url

    with requests.get(url, stream=True) as reader, open(file_path, "wb") as writer:
        total_size = int(reader.headers["Content-Length"])

        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, total=total_size, desc="Downloading: ") as progress:
            for chunk in reader.iter_content(chunk_size=1024*1024):
                stored_size = writer.write(chunk)
                progress.update(stored_size)


def lazy_download(url, file_path):
    if not os.path.exists(file_path):
        download(url, file_path)

def unzip(m):
    with zipfile.ZipFile(m, 'r') as zf:
        target_path = os.path.dirname(m)
        for member in tqdm(zf.infolist(), desc=f'Unzipping {m} into {target_path}'):
            zf.extract(member, target_path)


def lazy_unzip(pathToZipFile: str):
    if not os.path.isfile(pathToZipFile[:-4]):
        unzip(pathToZipFile)

