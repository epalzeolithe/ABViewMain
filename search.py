import os
import re

def get_last_two_insv_files(directory):
    pattern = re.compile(r"^VID_.*?(\d{3})\.insv$", re.IGNORECASE)

    files_with_index = []

    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            files_with_index.append((index, f))

    # Trier par index croissant
    files_with_index.sort(key=lambda x: x[0])

    if not files_with_index:
        return None, "none.insv"

    if len(files_with_index) == 1:
        last_file = files_with_index[0][1]
        return last_file, "none.insv"

    # Avant-dernier et dernier
    second_last = files_with_index[-2][1]
    last = files_with_index[-1][1]

    return second_last, last

def get_last_GPS_log_file(directory):
    pattern = re.compile(r"^LOG.*?(\d{5})\.txt$", re.IGNORECASE)

    files_with_index = []

    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            files_with_index.append((index, f))

    if not files_with_index:
        return "none.txt"

    # Tri par index croissant
    files_with_index.sort(key=lambda x: x[0])

    # Retourne le dernier fichier
    return files_with_index[-1][1]

print(get_last_two_insv_files("data/raw"))
print(get_last_GPS_log_file("data/raw"))