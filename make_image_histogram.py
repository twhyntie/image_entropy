#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

 Make an image histogram.

 See the README.md file and the GitHub wiki for more information.

 http://moedal.web.cern.ch

"""

# Import the code needed to manage files.
import os, glob

#...for parsing the arguments.
import argparse

#...for the logging.
import logging as lg

##...for file manipulation.
#from shutil import rmtree

#...for the plotting.
import matplotlib.pyplot as plt

#...for the image manipulation.
import matplotlib.image as mpimg

#...for the MATH.
import numpy as np

##...for the image scaling.
#import scipy.ndimage.interpolation as inter

# Load the LaTeX text plot libraries.
from matplotlib import rc

# Uncomment to use LaTeX for the plot text.
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)


if __name__ == "__main__":

    print("*")
    print("*===========================*")
    print("* Making an image histogram *")
    print("*===========================*")

    # Get the datafile path from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("dataPath",         help="Path to the input image.")
    parser.add_argument("outputPath",       help="Path to the output folder.")
#    parser.add_argument("--subject-width",  help="The desired subject image width [pixels].",  default=128, type=int)
#    parser.add_argument("--subject-height", help="The desired subject image height [pixels].", default=128, type=int)
    parser.add_argument("-v", "--verbose",  help="Increase output verbosity", action="store_true")
    args = parser.parse_args()

    ## The path to the image.
    data_path = args.dataPath
    #data_path = os.path.join(args.dataPath, "RAW/data")
    #
    if not os.path.exists(data_path):
        raise IOError("* ERROR: Unable to find image at '%s'." % (data_path))

    ## The output path.
    output_path = args.outputPath
    #output_path = "./"
    #output_path = os.path.join(args.dataPath, "SPL/data")
    #
    # Check if the output directory exists. If it doesn't, raise an error.
    if not os.path.isdir(output_path):
        raise IOError("* ERROR: '%s' output directory does not exist!" % (output_path))

#    ## The required width of the subject images [pixels].
#    SubjectWidth = args.subject_width
#
#    ## The required height of the split images [pixels].
#    SubjectHeight = args.subject_height

    # Set the logging level.
    if args.verbose:
        level=lg.DEBUG
    else:
        level=lg.INFO

    ## Log file path.
    log_file_path = os.path.join(output_path, 'log_make_image_histogram.log')

    # Configure the logging.
    lg.basicConfig(filename=log_file_path, filemode='w', level=level)

    print("*")
    print("* Image path        : '%s'" % (data_path))
    print("* Writing to        : '%s'" % (output_path))
    print("*")

    lg.info(" *=========================*")
    lg.info(" * Making image histograms *")
    lg.info(" *=========================*")
    lg.info(" *")
    lg.info(" * Data path         : '%s'" % (data_path))
    lg.info(" * Writing to        : '%s'" % (output_path))


#    lg.info(" *")
#    lg.info(" * Required subject size : %d x %d" % (SubjectWidth, SubjectHeight))
    lg.info(" *")

    ## The image as a NumPy array.
    img = mpimg.imread(data_path)

    #u_min = 0

    ## The original image width [pixels].
    ImageWidth = img.shape[1]

    #v_min = 0

    ## The original image height [pixels].
    ImageHeight = img.shape[0]

    ## The total number of pixels [pixels].
    NumberOfPixels = ImageWidth * ImageHeight

    lg.info(" * Current image                    : %s" % (data_path))
    lg.info(" *")
    lg.info(" * N_x (ImageWidth)       = % 6d [pixels]" % (ImageWidth) )
    lg.info(" * N_y (ImageHeight)      = % 6d [pixels]" % (ImageHeight))
    lg.info(" *")
    lg.info(" * N = N_x * N_y          = % 6d [pixels]" % (NumberOfPixels))
    lg.info(" *")

    # Get the three image channels.

    #
    # THE RED CHANNEL
    #

    ## The red (R) channel image.
    img_R = 255 * img[:,:,0]

    # Create the figure for the original image.
    mpimg.imsave(os.path.join(output_path, "img_R.png"), img_R, cmap='gray', vmin=0.0, vmax=255.0)

    ### Test array for the red channel.
    #img_R = np.array([[20,20],[20,210]])

    ## The width of the red channel image.
    ImageWidth_R = img_R.shape[1]

    ## The height of the red channel image.
    ImageHeight_R = img_R.shape[0]

    ## The minimum value in the red channel.
    min_r = np.amin(img_R)

    ## The maximum value in the red channel.
    max_r = np.amax(img_R)

    ## The number of pixels in the red channel image.
    NumberOfPixels_R = ImageHeight_R * ImageWidth_R

    lg.info(" *")
    lg.info(" * The Red Channel")
    lg.info(" *")
    lg.info(" * (min., max.) = (% 3d, % 3d)" % (min_r, max_r))
    lg.info(" *")
    lg.info(" * N_x_R (ImageWidth_R)         = % 6d [pixels]" % (ImageWidth_R) )
    lg.info(" * N_y_R (ImageHeight_R)        = % 6d [pixels]" % (ImageHeight_R))
    lg.info(" *")
    lg.info(" * N_R = N_x_R * N_y_R          = % 6d [pixels]" % (NumberOfPixels_R))
    lg.info(" *")


    # The red channel histogram and bins.
    raw_hist_R, raw_bins_R = np.histogram(img_R, bins=256, range=(0.0, 256.0))

    ## The (non-zero) histogram entries for n_{r'}.
    hist_R = []

    ## The left-most bin edge for r'.
    bins_R = []

    ## A dictionary of {alphabet_b:p_b'}.
    probs_r = {}

    # Loop over the histogrammed values.
    for i, val in enumerate(raw_hist_R):

        # Add to the probability dirctionary.
        probs_r[i] = float(val)/float(NumberOfPixels_R)

        if val > 0:
            hist_R.append(val)
            bins_R.append(raw_bins_R[i])

    # Convert the lists to arrays.
    hist_R = np.array(hist_R)
    bins_R = np.array(bins_R)


    # Create the image histogram figure for the red channel
    #-------------------------------------------------------
    plt.close('all')

    ## The property histogram plot.
    plot_R = plt.figure(101, figsize=(5.0, 3.0), dpi=150, facecolor='w', edgecolor='w')

    # Adjust the position of the axes.
    plot_R.subplots_adjust(bottom=0.17, left=0.15)

    ## The plot axes.
    pax_R = plot_R.add_subplot(111)

    # y axis
    plt.ylabel('%s' % ("$n_{r'}$"))

    # x axis
    plt.xlabel('%s' % ("$r'$"))

    # Add a grid.
    plt.grid(1)

    ## The x minimum.
    x_min = 0

    ## The x maximum.
    x_max = 256

    ## The bin width.
    bin_width_R = 1.0 * (bins_R[1] - bins_R[0])

    ## The bars for the red channel histogram.
    #
    # Note: we're using a NumPy histogram then plotting with a bar chart.
    bars_R = plt.bar(bins_R, hist_R, width=bin_width_R, linewidth=0, color='#880000')

    ## The maximum y value of the n_{r'} values.
    max_y_r = pax_R.get_ylim()[1]

    # Add lines for the min. and max. values.
    plt.vlines([min_r, max_r], 0, max_y_r, linestyle='dashed')

    # Set the x limits to the min. and max. value of r.
    pax_R.set_xlim([x_min, x_max])

    # Set the y limits to the min. and max. value of r.
    pax_R.set_ylim([0, max_y_r])

    # Save the figure.
    plot_R.savefig("%s/%s.png" % (output_path, "im_hist_R"))

    #
    # THE BLUE CHANNEL
    #

    ## The blue (B) channel image.
    img_B = 255 * img[:,:,2]

    # Create the figure for the original image.
    mpimg.imsave(os.path.join(output_path, "img_B.png"), img_B, cmap='gray', vmin=0.0, vmax=255.0)

    ### Test array for the blue channel.
    #img_B = np.array([[20,20],[20,210]])

    ## The width of the blue channel image.
    ImageWidth_B = img_B.shape[1]

    ## The height of the blue channel image.
    ImageHeight_B = img_B.shape[0]

    ## The number of pixels in the blue channel image.
    NumberOfPixels_B = ImageHeight_B * ImageWidth_B

    ## The minimum value in the blue channel.
    min_b = np.amin(img_B)

    ## The maximum value in the blue channel.
    max_b = np.amax(img_B)

    lg.info(" *")
    lg.info(" * The Blue Channel")
    lg.info(" *")
    lg.info(" * (min., max.) = (% 3d, % 3d)" % (min_b, max_b))
    lg.info(" *")
    lg.info(" * N_x_B (ImageWidth_B)         = % 6d [pixels]" % (ImageWidth_B) )
    lg.info(" * N_y_B (ImageHeight_B)        = % 6d [pixels]" % (ImageHeight_B))
    lg.info(" *")
    lg.info(" * N_B = N_x_B * N_y_B          = % 6d [pixels]" % (NumberOfPixels_B))
    lg.info(" *")

    # The blue channel histogram and bins.
    raw_hist_B, raw_bins_B = np.histogram(img_B, bins=256, range=(0.0, 256.0))

    ## The (non-zero) histogram entries for n_{b'}.
    hist_B = []

    ## The left-most bin edge for b'.
    bins_B = []

    ## A dictionary of {alphabet_b:p_b'}.
    probs_b = {}

    # Loop over the histogrammed values.
    for i, val in enumerate(raw_hist_B):

        # Add to the probability dirctionary.
        probs_b[i] = float(val)/float(NumberOfPixels_B)

        # This make the histogram plot nicer - no points for zero entries.
        if val > 0:
            hist_B.append(val)
            bins_B.append(raw_bins_B[i])

    #lg.info(probs_b)

    # Convert the lists to arrays.
    hist_B = np.array(hist_B)
    bins_B = np.array(bins_B)

    # Create the image histogram figure for the blue channel
    #--------------------------------------------------------
    plt.close('all')

    ## The property histogram plot.
    plot_B = plt.figure(102, figsize=(5.0, 3.0), dpi=150, facecolor='w', edgecolor='w')

    # Adjust the position of the axes.
    plot_B.subplots_adjust(bottom=0.17, left=0.15)

    ## The plot axes.
    pax_B = plot_B.add_subplot(111)

    # y axis
    plt.ylabel('%s' % ("$n_{b'}$"))

    # x axis
    plt.xlabel('%s' % ("$b'$"))

    # Add a grid.
    plt.grid(1)

    ## The bin width (blue channel).
    bin_width_B = 1.0 #* (bins_B[1] - bins_B[0])

    ## The bars for the blue channel histogram.
    #
    # Note: we're using a NumPy histogram then plotting with a bar chart.
    bars_B = plt.bar(bins_B, hist_B, width=bin_width_B, linewidth=0, color='#000088')

    ## The maximum y value of the n_{b'} values.
    max_y_b = pax_B.get_ylim()[1]

    # Add lines for the min. and max. values.
    plt.vlines([min_b, max_b], 0, max_y_b, linestyle='dashed')

    # Set the blue channel x axis limits.
    pax_B.set_xlim([x_min, x_max])

    # Set the y limits to the min. and max. value of r.
    pax_B.set_ylim([0, max_y_b])


    # Save the figure.
    plot_B.savefig("%s/%s.png" % (output_path, "im_hist_B"))


    #
    # CHECKS
    #
    lg.info(" *--------")
    lg.info(" * CHECKS ")
    lg.info(" *--------")

    # Check that we have the right number of pixels.
    #
    NumberOfPixels_R_from_hist = np.sum(hist_R)
    NumberOfPixels_B_from_hist = np.sum(hist_B)

    if NumberOfPixels_R_from_hist != NumberOfPixels_R:
        raise IOError("* ERROR! NumberOfPixels_R_from_hist != NumberOfPixels_R")
    if NumberOfPixels_B_from_hist != NumberOfPixels_B:
        raise IOError("* ERROR! NumberOfPixels_B_from_hist != NumberOfPixels_B")

    lg.info(" *")
    lg.info(" * NumberOfPixels_R_from_hist = %d" % (NumberOfPixels_R_from_hist))
    lg.info(" * NumberOfPixels_B_from_hist = %d" % (NumberOfPixels_B_from_hist))
    lg.info(" *")

    ## Sum of the red channel probabilties.
    sum_probs_r = sum(probs_r.values())

    ## Sum of the blue channel probabilties.
    sum_probs_b = sum(probs_b.values())

    lg.info(" * Sum {probs_b} = %f" % (sum_probs_b))
    lg.info(" * Sum {probs_r} = %f" % (sum_probs_r))
    lg.info(" *")


    # Entropy calculations
    #----------------------
    lg.info(" *--------------")
    lg.info(" * CALCULATIONS ")
    lg.info(" *--------------")
    lg.info(" *")

    ## The Shannon entropy of the red channel (brute force).
    entropy_R = 0.0

    # Loop over every pixel in the array.
    for i in range(ImageWidth_R):
        for j in range(ImageHeight_R):

            ## The value of r for this pixel.
            my_r = int(img_R[i][j])

            # Look up the probability of this value from the dictionary.
            prob_r_dash = probs_r[my_r]

            ## The Shannon information content of this outcome.
            info_content_r_dash = -1.0 * np.log2(prob_r_dash)

            ## The summed term in the entropy calculation.
            prob_r_d_times_info = prob_r_dash * info_content_r_dash

            # Add this to the total entropy of the image.
            entropy_R += prob_r_d_times_info

            #lg.info(" * (% 3d, % 3d): r = % 3d, Pr(r=r') = %f, -log_2(Pr) = %f -> %f" % (i, j, my_r, prob_r_dash, info_content_r_dash, prob_r_d_times_info))

    ## The Shannon entropy of the red channel (histogram).
    entropy_R_from_hist = 0.0

    # Calculate entropy from the histogram alone.
    for p_r in probs_r.values():
        if p_r > 0.0:
            entropy_R_from_hist += (p_r * NumberOfPixels_R) * (-1.0) * (p_r * np.log2(p_r))

    lg.info(" * Entropy content of R channel (brute force)    : %f" % (entropy_R))
    lg.info(" * Entropy content of R channel (from histogram) : %f" % (entropy_R_from_hist))
    lg.info(" *")

    ## The Shannon entropy of the blue channel (brute force).
    entropy_B = 0.0

    # Loop over every pixel in the array.
    for i in range(ImageWidth_B):
        for j in range(ImageHeight_B):

            ## The value of b for this pixel.
            my_b = int(img_B[i][j])

            # Look up the probability of this value from the dictionary.
            prob_b_dash = probs_b[my_b]

            ## The Shannon information content of this outcome.
            info_content_b_dash = -1.0 * np.log2(prob_b_dash)

            ## The summed term in the entropy calculation.
            prob_b_d_times_info = prob_b_dash * info_content_b_dash

            # Add this to the total entropy of the image.
            entropy_B += prob_b_d_times_info

            #lg.info(" * (% 3d, % 3d): r = % 3d, Pr(r=r') = %f, -log_2(Pr) = %f -> %f" % (i, j, my_b, prob_b_dash, info_content_b_dash, prob_b_d_times_info))

    ## The Shannon entropy of the blue channel (histogram).
    entropy_B_from_hist = 0.0

    # Calculate entropy from the histogram alone.
    for p_b in probs_b.values():
        if p_b > 0.0:
            entropy_B_from_hist += (p_b * NumberOfPixels_B) * (-1.0) * (p_b * np.log2(p_b))

    lg.info(" * Entropy content of B channel (brute force)    : %f" % (entropy_B))
    lg.info(" * Entropy content of B channel (from histogram) : %f" % (entropy_B_from_hist))
    lg.info(" *")


    ### The green (G) channel image.
    #img_G = 255 * img[:,:,1]
    #
    #lg.info(" * Green channel : % 9.4f - % 9.4f" % (np.amin(img_G), np.amax(img_G)))
