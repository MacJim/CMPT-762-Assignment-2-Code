import matplotlib.pyplot as plt

from epoch_logger import read_epoch_details_from_file


def main(train_log_filename: str, validation_log_filename: str, plot_save_filename: str):
    plt.ioff()
    fig = plt.figure(figsize=(15, 8))

    train_log = read_epoch_details_from_file(train_log_filename)
    validation_log = read_epoch_details_from_file(validation_log_filename)

    train_epochs = [value[0] for value in train_log]
    train_losses = [value[5] for value in train_log]
    validation_epochs = [value[0] for value in validation_log]
    validation_losses = [value[3] for value in validation_log]

    plt.subplot(2, 1, 1)
    plt.ylabel('Train loss')
    plt.plot(train_epochs, train_losses, 'k-')
    plt.title('train loss and val accuracy')
    plt.xticks(train_epochs)
    # Label visibility source: https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if ((i + 1) % 50):
            label.set_visible(False)
    plt.grid(True, axis="y")

    plt.subplot(2, 1, 2)
    plt.plot(validation_epochs, validation_losses, 'b-')
    plt.ylabel('Val accuracy')
    plt.xlabel('Epochs')
    plt.xticks(validation_epochs)
    for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if ((i + 1) % 50):
            label.set_visible(False)
    plt.grid(True, axis="y")

    plt.savefig(plot_save_filename, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main("checkpoints/train_log.csv", "checkpoints/validation_log.csv", "checkpoints/plot.png")
