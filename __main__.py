import customtkinter as ctk
from pathlib import Path
from search_frame import SearchFrame
from discover_frame import DiscoverFrame

class MusicDiscoveryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Set appearance and theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Window config
        self.title("Music Discovery")
        self.minsize(1480, 720)
        self.resizable(True, True)

        # Assets path
        self.assets_path = Path(__file__).parent / "assets"

        # Initialize frames
        self.search_frame = SearchFrame(
            master=self,
            assets_path=self.assets_path,
        )

        self.discover_frame = DiscoverFrame(
            master=self,
            assets_path=self.assets_path,
        )

        # Stack the frames
        for frame in (self.search_frame, self.discover_frame):
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Start with search frame
        self.show_search_frame()

    def show_search_frame(self):
        self.search_frame.lift()

    def show_discover_frame(self):
        self.discover_frame.lift()

# Example usage
if __name__ == "__main__":
    app = MusicDiscoveryApp()
    app.mainloop()
