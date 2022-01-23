import os
import imageio
import matplotlib.pyplot as plt

def save(drawframe, frames, name, transparency = False, remove = True):
    filenames = []
    for i in range(frames):
        # plot frame
        drawframe(i)

        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)

        # save frame
        plt.savefig(filename, transparency = transparency)
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