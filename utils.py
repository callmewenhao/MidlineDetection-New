import matplotlib.pyplot as plt


def show_landmarks(image, x, y):
    """Show image with landmarks"""
    plt.figure("image")
    plt.imshow(image, cmap='gray')
    plt.scatter(x, y, s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()
