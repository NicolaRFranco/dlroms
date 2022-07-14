import os
import imageio
import matplotlib.pyplot as plt

def save(drawframe, frames, name, remove = True):
    """Constructs a GIF given a way to plot each frame.
    
    Input
        drawframe       (function)      Function that specifies how to plot each frame. It should have a single argument,
                                        that being the number of the current frame.
        frames          (int)           Total number of frames.
        name            (str)           Path where to save the GIF file.
        transparency    (bool)          Whether to impose or not a transparent background. Defaults to False.
        remove          (bool)          Whether to leave on the disk or not the files corresponding to each frame.
                                        Defaults to True.
    """
    filenames = []
    for i in range(frames):
        # plot frame
        drawframe(i)

        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)

        # save frame
        plt.savefig(filename)
        plt.close()
    # build gif
    with imageio.get_writer(name + '.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    if(remove):
        for filename in set(filenames):
            os.remove(filename)
