# Written by: Nicola Rares Franco, Ph.D. (MOX, Department of Mathematics, Politecnico di Milano)
# 
# Scientific articles based on this Python package:
#
# [1] Franco et al., Mathematics of Computation (2023).
#     A deep learning approach to reduced order modelling of parameter dependent partial differential equations.
#     DOI: https://doi.org/10.1090/mcom/3781.
#
# [2] Franco et al., Neural Networks (2023).
#     Approximation bounds for convolutional neural networks in operator learning.
#     DOI: https://doi.org/10.1016/j.neunet.2023.01.029
#
# [3] Franco et al., Journal of Scientific Computing (2023). 
#     Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces.
#     DOI: https://doi.org/10.1007/s10915-023-02331-1
#
# [4] Vitullo, Colombo, Franco et al., Finite Elements in Analysis and Design (2024).
#     Nonlinear model order reduction for problems with microstructure using mesh informed neural networks.
#     DOI: https://doi.org/10.1016/j.finel.2023.104068
#
# Please cite the Author if you use this code for your work/research.

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
        filename = 'temp-gif-frame%d.png' % i
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
