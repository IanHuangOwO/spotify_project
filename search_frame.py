from pathlib import Path
import customtkinter as ctk
from customtkinter import CTkImage
import tkinter as tk
import csv
import threading
import os
from PIL import Image, ImageTk
from spotify_api import SpotifyAPI
import webbrowser
from Recom_Song_API import get_song_API
from dotenv import load_dotenv
import yaml

class SearchFrame(ctk.CTkFrame):
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
        
        self.song_selected = None
        self.song_data = []
        self.typing_timer = None
        self.latest_query = ""

        # Load CSV in a separate thread
        threading.Thread(target=self.load_csv_data, daemon=True).start()
        threading.Thread(target=self.load_yaml_data, daemon=True).start()
        
        self.setup_ui()
        self.bind("<Configure>", self.resize_background)

    def relative_to_assets(self, path: str) -> Path:
        return self.assets_path / Path(path)

    def load_csv_data(self):
        try:
            filter_data_path = self.assets_path / "data/filter_data.csv"
            with open(filter_data_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.song_data = list(reader)
            
            universe_avg_path = self.assets_path / "data/universe_avg_data.csv"
            with open(universe_avg_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.universe_avg_data = list(reader)
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    def load_yaml_data(self):
        try:
            yaml_path = self.assets_path / "config.yaml"
            with open(yaml_path, "r") as file:
                data = yaml.safe_load(file)
                self.country_tags = data["countries"]
                self.country_language = data['languages']
                self.feature_columns = data["feature_columns"]
        except Exception as e:
            print(f"Error loading YMAL: {e}")

    def setup_ui(self):
        # Canvas
        self.canvas = ctk.CTkCanvas(self, bg="#0F1920", width=1280, height=720, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.original_bg_image = Image.open(self.relative_to_assets("defaults/background.png"))
        self.bg_image_tk = ImageTk.PhotoImage(self.original_bg_image)
        self.canvas_bg = self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image_tk)
        
        # Rounded panel
        self.rounded_panel = ctk.CTkFrame(self, 
                                          width=380, height=700, corner_radius=25, fg_color="#0F1920", border_color="#5A5A5A", background_corner_colors= ("#0F1920", "#0F1920", "#0F1920", "#0F1920") , border_width=2)
        self.rounded_panel.place(relx=0.5, rely=0.5, anchor="center")

        # Title
        self.title_label = ctk.CTkLabel(self.rounded_panel, 
                                        text="Discover\nNew\nSong", justify="center", font=("Inter", 40, "bold"), text_color="#A77B1D")
        self.title_label.place(relx=0.5, y=50, anchor="n")
        
        # Create the CTkOptionMenu
        country_names = list(self.country_tags.keys())
        
        self.option = ctk.CTkOptionMenu(self.rounded_panel, 
                                     width=250, height=30, 
                                     bg_color="#0F1920", fg_color="#353535", text_color = "#999999", button_color= "#353535", button_hover_color = "#505050", corner_radius=35, values=country_names)
        self.option.place(relx=0.5, y=230, anchor="n")
        
        self.checkbox = ctk.CTkCheckBox(master=self.rounded_panel,
                                        checkbox_width=15, checkbox_height=15, text="Only local language song?", font=("Inter", 14), border_width= 1)
        self.checkbox.place(relx=0.5, y=263, anchor="n")
        
        # Search Entry
        self.search_entry = ctk.CTkEntry(self.rounded_panel, 
                                         width=250, height=30, corner_radius=35, border_width = 0, placeholder_text="Search your song")
        self.search_entry.place(relx=0.5, y=290, anchor="n")
        self.search_entry.bind("<KeyRelease>", self.on_entry_keyrelease)
        
        # Dropdown
        self.dropdown = tk.Toplevel(self)
        self.dropdown.withdraw()
        self.dropdown.overrideredirect(True)
        self.dropdown.lift()
        
        self.dropdown_listbox = tk.Listbox(self.dropdown, 
                                           bg="#454545", fg="#999999", activestyle='dotbox', font=("Inter", 10), highlightthickness=0)
        self.dropdown_listbox.pack(fill="both", expand=True)
        self.dropdown_listbox.bind("<<ListboxSelect>>", self.on_dropdown_select)

        # Album image
        self.album_image_label = ctk.CTkLabel(self.rounded_panel, text="")
        self.album_image_label.place(x=120, y=330)
        self.update_album_image()
        
        # Song info labels
        self.song_title_label = ctk.CTkLabel(self.rounded_panel, 
                                             text="Big Yellow Chicken", font=("Inter", 16, "bold"), text_color="#E4E4E4")
        self.song_title_label.place(x=120, y=480)

        self.artist_label = ctk.CTkLabel(self.rounded_panel, 
                                         text="Yellow Chicken", font=("Inter", 10), text_color="#AAAAAA")
        self.artist_label.place(x=120, y=500)
        
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
        self.youtube_button.place(x=150, y=540, anchor="n")
        
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
        self.spotify_button.place(x=190, y=540, anchor="n")
        
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
        self.apple_button.place(x=230, y=540, anchor="n")
        
        # Discover Button
        self.button = ctk.CTkButton(
            self.rounded_panel,
            text="Discover",
            width=175,
            height=40,
            font=("Inter", 14, "bold"),
            fg_color="#396C34",
            text_color="#FFFFFF",
            hover_color="#4EA845",
            corner_radius=35,
            command=self.on_discover_click
        )
        self.button.place(relx=0.5, y=600, anchor="n")
    
    def resize_background(self, event):
        new_width = event.width
        new_height = event.height

        resized = self.original_bg_image.resize((new_width, new_height), Image.LANCZOS)
        self.bg_image_tk = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self.canvas_bg, image=self.bg_image_tk)
    
    def on_entry_keyrelease(self, event):
        if self.typing_timer:
            self.typing_timer.cancel()
            
        self.latest_query = self.search_entry.get()
        
        self.typing_timer = threading.Timer(0.5, self.perform_search)
        self.typing_timer.start()

    def perform_search(self):
        query = self.latest_query.lower()
        matches = [row for row in self.song_data if query in row["name"].lower() or query in row["artists"].lower()]
        
        if matches:
            self.dropdown_listbox.delete(0, tk.END)
            for row in matches[:50]:
                self.dropdown_listbox.insert(tk.END, f"{row['name']} - {row['artists']}")
            self.show_dropdown()
        else:
            self.dropdown.withdraw()

    def show_dropdown(self):
        x = self.search_entry.winfo_rootx()
        y = self.search_entry.winfo_rooty() + self.search_entry.winfo_height()
        self.dropdown.geometry(f"280x200+{x + 15}+{y + 3}")
        self.dropdown.deiconify()

    def on_dropdown_select(self, event):
        if not self.dropdown_listbox.curselection():
            return

        index = self.dropdown_listbox.curselection()[0]
        selected_text = self.dropdown_listbox.get(index)
        selected_name = selected_text.split(" - ")[0]

        self.song_selected = next((s for s in self.song_data if s["name"] == selected_name), None)
        if not self.song_selected:
            return
        
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, self.song_selected["name"])
        self.dropdown.withdraw()
        
        try:
            self.update_song_info()
        except Exception as e:
            print(f"Spotify API error: {e}")

    def update_song_info(self):
        track_info = self.spotify.get_track_info(self.song_selected["id"])
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
        
    def on_discover_click(self):
        self.setup_recommanded_data()
        self.parent.show_discover_frame()
        
    def setup_recommanded_data(self):
        import numpy as np
        from pandas import DataFrame
        
        if self.song_data is None:
            return
        
        current_country_tag = self.country_tags[self.option.get()]
        country_selected = next((item for item in self.universe_avg_data if item['country'] == current_country_tag), None)
        
        input_feature = np.array([float(self.song_selected[key]) for key in self.feature_columns])
        country_feature = np.array([float(country_selected[key]) for key in self.feature_columns])

        features = 0.8 * input_feature + 0.2 * country_feature
        
        current_language = self.country_language.get(self.option.get(), None) if self.checkbox.get() else None
        
        results = get_song_API(features_df= DataFrame([features], columns=self.feature_columns), language = current_language)
        
        self.parent.discover_frame.original_id = self.song_selected["id"]
        self.parent.discover_frame.update_song_info(self.song_selected["id"])
        
        id_fields = ["first_id", "second_id", "third_id", "opposite_id"]
        
        for i, attr in enumerate(id_fields):
            if len(results) > i:
                value = results[i][0] if (i != 0 or results[i][1] > 0.8) else None
                setattr(self.parent.discover_frame, attr, value)
            else:
                setattr(self.parent.discover_frame, attr, None)

        
        