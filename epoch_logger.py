import csv
import os
import typing


def log_epoch_details_to_file(epoch: int, total_images_count: int, correct_images_count: int, correct_percentage: float, total_loss: float, average_loss: float, time: float, filename: str):
    if (os.path.isdir(filename)):
        raise FileExistsError(f"Target file `{filename} is a directory.")

    if (not os.path.exists(filename)):
        # Create the log file.
        with open(filename, "w") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["Epoch", "Total Count", "Correct Count", "Correct Percentage", "Total Loss", "Average Loss", "Time"])

    with open(filename, "a") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow([epoch, total_images_count, correct_images_count, correct_percentage, total_loss, average_loss, time])


def read_epoch_details_from_file(filename: str) -> typing.List[typing.List[typing.Any]]:
    if (not os.path.isfile(filename)):
        raise FileNotFoundError(f"`{filename}` is not a file.")

    with open(filename, "r") as f:
        r = csv.reader(f)
        return_value = list(r)

    if return_value:
        # Remove the titles.
        return_value = return_value[1:]
        # Convert to numbers.
        return_value = [[int(entry[0]), int(entry[1]), int(entry[2]), float(entry[3]), float(entry[4]), float(entry[5]), float(entry[6])] for entry in return_value]

    return return_value
