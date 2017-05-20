# Import the os module, for the os.walk function
import os

# Import the fnmatch module, for filtering au files
import fnmatch

# Import the AudioSegment module from pydub, for converting au files to uncompressed wav format
from pydub import AudioSegment

# Set the directory you want to start from
genres_dir = '/home/michael/Documents/mgc/genres'

genre_folders = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
processed_dir = genres_dir[: genres_dir.rindex('/') + 1] + "/processed_genres/"

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Make directories for processed dataset
for folder in genre_folders:
    if not os.path.exists(processed_dir + folder):
        os.mkdir(os.path.join(processed_dir, folder))

# Create a processed dataset
for dir_name, subdir_list, file_list in os.walk(genres_dir):
    print('Found directory: %s' % dir_name)
    for fname in fnmatch.filter(file_list, '*.au'):
        filepath = dir_name + "/" + fname
        processed_filepath = filepath.replace("genres", "processed_genres")
        au_version = AudioSegment.from_file(filepath, "au")
        au_version.export(processed_filepath.replace("au", "wav"), format="wav")
