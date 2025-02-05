import os
import random
import time
import pygame
from pathlib import Path

def list_audio_files():
    """List all audio files in the audio directory."""
    audio_dir = Path("audio")
    if not audio_dir.exists():
        print("Error: audio directory not found!")
        return []
    
    audio_files = []
    supported_formats = {'.mp3', '.wav', '.ogg'}
    
    for file in audio_dir.iterdir():
        if file.suffix.lower() in supported_formats:
            audio_files.append(file)
    
    return audio_files

def play_audio_files(audio_files):
    """Play audio files on shuffle and repeat."""
    if not audio_files:
        print("No audio files found!")
        return
    
    pygame.mixer.init()
    print("\nInitialized audio playback. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Shuffle the playlist
            random.shuffle(audio_files)
            print("\nNew shuffle order:")
            for i, file in enumerate(audio_files, 1):
                print(f"{i}. {file.name}")
            
            # Play each file in the shuffled order
            for file in audio_files:
                print(f"\nNow playing: {file.name}")
                pygame.mixer.music.load(str(file))
                pygame.mixer.music.play()
                
                # Wait for the current song to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping audio playback...")
        pygame.mixer.quit()
        print("Goodbye!")

def main():
    print("Audio Test Script")
    print("================")
    
    # List all audio files
    print("\nScanning audio directory...")
    audio_files = list_audio_files()
    
    if audio_files:
        print(f"\nFound {len(audio_files)} audio file(s):")
        for i, file in enumerate(audio_files, 1):
            print(f"{i}. {file.name}")
        
        # Start playback
        play_audio_files(audio_files)
    else:
        print("Please add some audio files to the audio directory.")

if __name__ == "__main__":
    main() 