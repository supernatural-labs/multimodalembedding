This simple project allows you to search a video using natural language. It does this by sampling 1 frame per second and creating an embedding of it using Google's Vertex API, storing the results in chromadb. It then embeds the user's query with the same API and finds the frame that's the best match and opens VLC player to that exact spot.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Examples:

`python3 predict_request_gapic.py --project super-399400 --text back -video_file </path/to/folder>`
