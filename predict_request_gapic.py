# Copyright 2023 Google LLC.
# SPDX-License-Identifier: Apache-2.0
from absl import app
from absl import flags
import base64
# Need to do pip install google-cloud-aiplatform for the following two imports.
# Also run: gcloud auth application-default login.
from google.cloud import aiplatform
from google.protobuf import struct_pb2
import sys
import time
import typing
import numpy as np
import os
import chromadb
from tqdm import tqdm
import hashlib
import imageio.v3 as iio
import subprocess

_IMAGE_DIR = flags.DEFINE_string('image_dir', None, 'Image directory')
_IMAGE_FILE = flags.DEFINE_string('image_file', None, 'Image filename')
_VIDEO_FILE = flags.DEFINE_string('video_file', None, 'Video filename')
_TEXT = flags.DEFINE_string('text', None, 'Text to input')
_PROJECT = flags.DEFINE_string('project', None, 'Project id')

# Inspired from https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple.
class EmbeddingResponse(typing.NamedTuple):
  text_embedding: typing.Sequence[float]
  image_embedding: typing.Sequence[float]


class EmbeddingPredictionClient:
  """Wrapper around Prediction Service Client."""
  def __init__(self, project : str,
    location : str = "us-central1",
    api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
    client_options = {"api_endpoint": api_regional_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)  
    self.location = location
    self.project = project
    self.chroma_client = chromadb.PersistentClient(path='chroma.db')
    self.collection = self.chroma_client.get_or_create_collection('images')

  def get_embedding(self, text : str = None, image_bytes : bytes = None):
    if not text and not image_bytes:
      raise ValueError('At least one of text or image_bytes must be specified.')

    instance = struct_pb2.Struct()
    if text:
      instance.fields['text'].string_value = text

    if image_bytes:
      encoded_content = base64.b64encode(image_bytes).decode("utf-8")
      image_struct = instance.fields['image'].struct_value
      image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

    instances = [instance]
    endpoint = (f"projects/{self.project}/locations/{self.location}"
      "/publishers/google/models/multimodalembedding@001")
    response = self.client.predict(endpoint=endpoint, instances=instances)

    text_embedding = None
    if text:    
      text_emb_value = response.predictions[0]['textEmbedding']
      text_embedding = [v for v in text_emb_value]

    image_embedding = None
    if image_bytes:    
      image_emb_value = response.predictions[0]['imageEmbedding']
      image_embedding = [v for v in image_emb_value]

    return EmbeddingResponse(
      text_embedding=text_embedding,
      image_embedding=image_embedding)
        

  def load_images(self, image_dir):
    '''Loads images from a directory., and adds each to a chroma db.'''
    h = hashlib.new("md5")
    for filename in tqdm(os.listdir(image_dir)):
      if filename.endswith('.jpg'):
        image_path = os.path.join(image_dir, filename)
        with open(image_path, "rb") as f:
          image_file_contents = f.read()
          h.update(image_file_contents)
          id = h.hexdigest()
          results = self.collection.get(ids=[id])
          if len(results['ids']) == 0:
            print('Adding image: ', image_path)
            response = self.get_embedding(image_bytes=image_file_contents)

            # # Add to chroma db.
            self.collection.add(
                embeddings=[response.image_embedding],
                documents=[image_path],
                ids=[id]
            )

  def load_video(self, video_file):
    '''Loads images from a video, and adds each to a chroma db.'''
    h = hashlib.new("md5")
    '''get frame rate of video file'''
    fps = iio.immeta(video_file)['fps']
    for idx, frame in tqdm(enumerate(iio.imiter(video_file))):
      if (idx % 30) != 0:
        continue
      iio.imwrite(f"extracted_images/frame{idx:03d}.jpg", frame)
      # convert frame to bytes
      image_contents = iio.imwrite("<bytes>", frame, extension=".png")
      h.update(image_contents)
      id = h.hexdigest()
      results = self.collection.get(ids=[id])
      if len(results['ids']) == 0:
        response = self.get_embedding(image_bytes=image_contents)

        # # Add to chroma db.
        self.collection.add(
            embeddings=[response.image_embedding],
            documents=[f"extracted_images/frame{idx:03d}.jpg"],
            metadatas=[{"seek_time": idx / fps , "file": video_file}],
            ids=[f"{idx:03d}"]
        )

def main(argv):
  client = EmbeddingPredictionClient(project=_PROJECT.value)
  
  start = time.time()
  if _IMAGE_DIR.value:
    client.load_images(_IMAGE_DIR.value)
  if _VIDEO_FILE.value:
    client.load_video(_VIDEO_FILE.value)
  response = client.get_embedding(text=_TEXT.value)
  end = time.time()

  matches = client.collection.query(
    query_embeddings=[response.text_embedding],
    n_results=2
  )
  print(matches)

  print('Time taken: ', end - start)
  print(matches['documents'][0])
  if matches['metadatas'][0][0]['file']:
    os.system(' '.join([
      'open',
      '-a /Applications/VLC.app/Contents/MacOS/VLC',
      f"{matches['metadatas'][0][0]['file']} --args --start-time {int(matches['metadatas'][0][0]['seek_time']) - 1}"
    ]))


if __name__ == "__main__":
    app.run(main)
