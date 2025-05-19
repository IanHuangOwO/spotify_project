from pathlib import Path
import customtkinter as ctk
from customtkinter import CTkImage
from PIL import Image, ImageTk
from spotify_api import SpotifyAPI
import webbrowser
import os
from dotenv import load_dotenv

class DiscoverFrame(ctk.CTkFrame):
    def __init__(self, master, assets_path: Path, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.parent = master
        self.assets_path = assets_path
        self.configure(fg_color="#2E2E2E")
        self.pack(fill="both", expand=True)
        
        # Spotify API init
        load_dotenv()
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.spotify = SpotifyAPI(client_id, client_secret)
        
        self.original_id = None
        self.first_id = None
        self.second_id = None
        self.third_id = None
        self.opposite_id = None
        
        self.setup_ui()
        self.bind("<Configure>", self.resize_background)
        
    def relative_to_assets(self, path: str) -> Path:
        return self.assets_path / Path(path)
    
    def setup_ui(self):
        # Canvas
        self.canvas = ctk.CTkCanvas(self, bg="#0F1920", width=1280, height=720, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.original_bg_image = Image.open(self.relative_to_assets("defaults/background.png"))
        self.bg_image_tk = ImageTk.PhotoImage(self.original_bg_image)
        self.canvas_bg = self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image_tk)
        
        self.original_button = ctk.CTkButton(
            self,
            text="Original\nSong",
            font=("Courier New", 20),
            width=40,
            height=20,
            fg_color = '#0c3337',
            hover_color = '#126E77',
            corner_radius= 40,
            background_corner_colors= ("#0c3337", "#0c3337", "#0c3337", "#0c3337", "#0c3337"),
            command= lambda: self.update_song_info(self.original_id)
        )
        self.original_button.place(relx=0.225, rely=0.47, anchor="n")
        
        self.first_button = ctk.CTkButton(
            self,
            text="First Song",
            font=("Courier New", 20),
            width=40,
            height=20,
            fg_color = "#0F1920",
            hover_color = "#1C2F3C",
            corner_radius= 40,
            background_corner_colors= ("#0F1920", "#0F1920", "#0F1920", "#0F1920", "#0F1920"),
            command= lambda: self.update_song_info(self.first_id),
        )
        self.first_button.place(relx=0.44, rely=0.43, anchor="n")
        
        self.second_button = ctk.CTkButton(
            self,
            text="Second Song",
            font=("Courier New", 20),
            width=40,
            height=20,
            fg_color = "#0F1920",
            hover_color = "#1C2F3C",
            corner_radius= 40,
            background_corner_colors= ("#0F1920", "#0F1920", "#0F1920", "#0F1920", "#0F1920"),
            command= lambda: self.update_song_info(self.second_id),
        )
        self.second_button.place(relx=0.4, rely=0.74, anchor="n")
        
        self.third_button = ctk.CTkButton(
            self,
            text="Third Song",
            font=("Courier New", 20),
            width=40,
            height=20,
            fg_color = "#0F1920",
            hover_color = "#1C2F3C",
            corner_radius= 40,
            background_corner_colors= ("#0F1920", "#0F1920", "#0F1920", "#0F1920", "#0F1920"),
            command= lambda: self.update_song_info(self.third_id),
        )
        self.third_button.place(relx=0.57, rely=0.535, anchor="n")
        
        self.opposite_button = ctk.CTkButton(
            self,
            text="Fourth Song",
            font=("Courier New", 20),
            width=40,
            height=20,
            fg_color = "#0F1920",
            hover_color = "#1C2F3C",
            corner_radius= 40,
            background_corner_colors= ("#0F1920", "#0F1920", "#0F1920", "#0F1920", "#0F1920"),
            command= lambda: self.update_song_info(self.opposite_id),
        )
        self.opposite_button.place(relx=0.67, rely=0.71, anchor="n")
        
        # Rounded panel
        self.rounded_panel = ctk.CTkFrame(self, 
                                          width=380, height=700, corner_radius=25, fg_color="#0F1920", border_color="#5A5A5A", background_corner_colors= ("#0F1920", "#0F1920", "#0F1920", "#0F1920") , border_width=2)
        self.rounded_panel.place(relx=0.87, rely=0.5, anchor="center")

        # Title
        self.title_label = ctk.CTkLabel(self.rounded_panel, 
                                        text="Original Song", justify="center", font=("Inter", 40, "bold"), text_color="#A77B1D")
        self.title_label.place(relx=0.5, y=70, anchor="n")

        # Album image
        self.album_image_label = ctk.CTkLabel(self.rounded_panel, text="")
        self.album_image_label.place(x=120, y=230)
        self.update_album_image()
        
        # Song info labels
        self.song_title_label = ctk.CTkLabel(self.rounded_panel, 
                                             text="Big Yellow Chicken", font=("Inter", 16, "bold"), text_color="#E4E4E4")
        self.song_title_label.place(x=120, y=380)

        self.artist_label = ctk.CTkLabel(self.rounded_panel, 
                                         text="Yellow Chicken", font=("Inter", 10), text_color="#AAAAAA")
        self.artist_label.place(x=120, y=400)
        
        # Listen Button
        youtube = Image.open(self.relative_to_assets("defaults/youtube_music.png")).resize((30, 30))
        self.youtube_image = ImageTk.PhotoImage(youtube)
        self.youtube_button = ctk.CTkButton(
            self.rounded_panel,
            image= self.youtube_image,
            text="",
            width=30,
            height=30,
            fg_color="#0F1920",
            hover_color="#505050",
        )
        self.youtube_button.place(x=150, y=450, anchor="n")
        
        spotify = Image.open(self.relative_to_assets("defaults/spotify_music.png")).resize((30, 30))
        self.spotify_image = ImageTk.PhotoImage(spotify)
        self.spotify_button = ctk.CTkButton(
            self.rounded_panel,
            image= self.spotify_image,
            text="",
            width=30,
            height=30,
            fg_color="#0F1920",
            hover_color="#505050",
            command=lambda: self.open_spotify_link(),
        )
        self.spotify_button.place(x=190, y=450, anchor="n")
        
        apple = Image.open(self.relative_to_assets("defaults/apple_music.png")).resize((30, 30))
        self.apple_image = ImageTk.PhotoImage(apple)
        self.apple_button = ctk.CTkButton(
            self.rounded_panel,
            image= self.apple_image,
            text="",
            width=30,
            height=30,
            fg_color="#0F1920",
            hover_color="#505050",
        )
        self.apple_button.place(x=230, y=450, anchor="n")
        
        # Discover Button
        self.button = ctk.CTkButton(
            self.rounded_panel,
            text="Try Again",
            width=175,
            height=40,
            font=("Inter", 14, "bold"),
            fg_color="#396C34",
            text_color="#FFFFFF",
            hover_color="#4EA845",
            corner_radius=35,
            command=lambda: self.on_tryagain_click(),
        )
        self.button.place(relx=0.5, y=580, anchor="n")
    
    def resize_background(self, event):
        new_width = event.width
        new_height = event.height

        resized = self.original_bg_image.resize((new_width, new_height), Image.LANCZOS)
        self.bg_image_tk = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self.canvas_bg, image=self.bg_image_tk)
    
    def update_song_info(self, song_id):
        track_info = self.spotify.get_track_info(song_id)
        self.song_title_label.configure(text=track_info['name'])
        self.artist_label.configure(text=track_info['artists'])
        self.update_album_image(track_info['image_url'])
        self.current_spotify_url = track_info["spotify_url"]
    
    def update_album_image(self, image_path = None):
        try: pil_image = self.spotify.get_album_image(image_path)
        except: pil_image = Image.open(self.relative_to_assets("defaults/album.png"))
        
        self.album_image = CTkImage(pil_image, size=(145, 145))
        self.album_image_label.configure(image=self.album_image)
        
    def open_spotify_link(self):
        if hasattr(self, 'current_spotify_url'):
            webbrowser.open(self.current_spotify_url)
        else:
            print("No Spotify URL available.")
        
    def on_tryagain_click(self):
        self.parent.show_search_frame()
