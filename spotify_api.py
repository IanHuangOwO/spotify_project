from PIL import Image
import requests
from io import BytesIO

class SpotifyAPI:
    def __init__(self, client_id, client_secret):
        self.token = self._get_token(client_id, client_secret)

    def _get_token(self, client_id, client_secret):
        resp = requests.post("https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret))
        resp.raise_for_status()
        return resp.json()["access_token"]

    def get_track_info(self, track_id, max_length = 20):
        headers = {"Authorization": f"Bearer {self.token}"}
        resp = requests.get(f"https://api.spotify.com/v1/tracks/{track_id}", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        if len(data["name"]) > max_length:
            data["name"] = data["name"][:max_length - 3] + "..."

        return {
            "name": data["name"],
            "artists": data["artists"][0]["name"],
            "image_url": data["album"]["images"][0]["url"],
            "spotify_url": data["external_urls"]["spotify"],
            "popularity": data["popularity"],
        }

    def get_album_image(self, url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
