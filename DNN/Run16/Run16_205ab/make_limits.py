#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
import json
from sklearn.metrics import auc

def make_hist_2D(data):
    """
    Generates and saves a 2D histogram / exclusion plot.

    Parameters
    ----------
    data : dict
        Dictionary loaded from 'DNN_results_run11.json'.
    """

    # Extract data points from the JSON `data`
    data_points = []
    for key, value in data.items():
        parts = key.split('_')
        mass = float(parts[0].replace('GeV', ''))  # Just take the number
        # Replace '1e-' with '-' and 'p' with '.' to get the exponent properly
        angle = float(parts[1].replace('1e-', '-').replace('p', '.'))
        highest_significance_entry = max(value['significance_list'], key=lambda x: x[0])
        highest_significance = highest_significance_entry[0]
        bdt_cut = highest_significance_entry[2]  # Third entry of the significance_list

        data_points.append((mass, angle, highest_significance, bdt_cut))

    # Prepare arrays for histogram
    masses = [float(d[0]) for d in data_points]
    angles = [float(d[1] * 2) for d in data_points]  # example: doubling or adjusting log(U^2)
    significances = [d[2] for d in data_points]
    bdt_cuts = [d[3] for d in data_points]

    # Print BDT cuts for debugging
    print(f"BDT cuts list: {bdt_cuts}")

    # Define histogram bins
    x_range = (0, 90)
    y_range = (-11, -3)
    x_bins = [4, 6, 14, 16, 24, 26, 34, 36, 44, 46, 54, 56, 64, 66, 74, 76, 84, 86]
    y_bins = np.arange(-10.75, -2.25, 0.5)
    n_bins = [x_bins, y_bins]

    # Define log scale normalization and color map
    norm = LogNorm(vmin=1e-5, vmax=5)
    cmap = plt.cm.RdPu

    # Create 2D histogram plot
    plt.figure(figsize=(8, 6))
    plt.hist2d(masses, angles, bins=n_bins, range=[x_range, y_range],
               weights=significances, cmap=cmap, norm=norm)

    # Accumulate data in hist for text-annotation later
    hist, edges = np.histogramdd((masses, angles), bins=n_bins,
                                 range=[x_range, y_range],
                                 weights=significances)
    x_edges, y_edges = edges

    # Add colorbar
    plt.colorbar(label='Z-significance at $10\\mathrm{fb}^{-1}$')

    # Prepare data for interpolation and contouring
    threshold = 2.0  # 2-sigma
    x_grid, y_grid = np.meshgrid(x_bins, y_bins)

    # Example cut & count data:
    data_points_cc = [
        ["10", -4.0, 52.876748861276596], ["10", -5.0, 15.128775602656333],
        ["10", -6.0, 4.4804573585508045], ["10", -7.0, 0.001475996382017464],
        ["10", -8.0, 0.08968425151707608], ["10", -9.0, 5.99037429924615e-08],
        ["10", -10.0, 1.4097477838218989e-05], ["20", -4.0, 36.2857862213626],
        ["20", -5.0, 10.884431421876464], ["20", -6.0, 2.4678689415467248],
        ["20", -7.0, 0.004537376980651871], ["20", -8.0, 0.058997010448686465],
        ["20", -9.0, 1.163004130746887e-06], ["20", -10.0, 3.352827837383747e-05],
        ["30", -4.0, 32.92212247543547], ["30", -5.0, 8.315052417103097],
        ["30", -6.0, 1.7459934119249754], ["30", -7.0, 0.0033373434392724114],
        ["30", -8.0, 0.029476881359299892], ["40", -4.0, 29.429788400754102],
        ["40", -5.0, 7.506664142646672], ["40", -6.0, 1.5869986460516172],
        ["40", -7.0, 0.0029276961561613734], ["40", -8.0, 0.02821520628356613],
        ["40", -10.0, 0], ["50", -4.0, 25.62219372240296],
        ["50", -5.0, 6.763465781950523], ["50", -6.0, 1.5570028443548596],
        ["50", -7.0, 0.003920231273620948], ["50", -8.0, 0.03817629371414823],
        ["50", -10.0, 0], ["60", -4.0, 18.866071450665995],
        ["60", -5.0, 4.839367338887292], ["60", -6.0, 1.0775817568201167],
        ["60", -7.0, 0.00230040705768753], ["60", -8.0, 0.022811124653626037],
        ["60", -9.0, 2.3613377004847655e-05], ["60", -10.0, 0.0002345337086266546],
        ["70", -4.0, 11.254577646746137], ["70", -5.0, 2.7412790459120084],
        ["70", -6.0, 0.5486607337910216], ["70", -7.0, 0.0008222118582734013],
        ["70", -8.0, 0.008246616737636249], ["70", -9.0, 8.055020468869638e-06],
        ["70", -10.0, 8.326387828664493e-05], ["80", -4.0, 0.6762462099536374],
        ["80", -5.0, 0.09792341746750198], ["80", -6.0, 0.010659487433272599],
        ["80", -7.0, 9.878873485179866e-06], ["80", -8.0, 0.00010780352635664964],
        ["80", -9.0, 1.508599479535031e-07], ["80", -10.0, 1.2065998272242434e-06]
    ]
    masses_cc = [float(d[0]) for d in data_points_cc]
    angles_cc = [float(d[1]) for d in data_points_cc]
    significances_cc = [d[2] for d in data_points_cc]

    # Interpolate the significance values from both methods
    z_interp = griddata((masses, angles), significances, (x_grid, y_grid),
                        method='linear', fill_value=0)
    z_cut_count = griddata((masses_cc, angles_cc), significances_cc,
                           (x_grid, y_grid), method='linear', fill_value=0)

    # DNN 2 sigma contour
    contour_2sigma = plt.contour(x_grid, y_grid, z_interp, levels=[threshold],
                                 colors='magenta', linestyles='dashed')
    # Cut&Count 2 sigma contour
    cc_2sigma = plt.contour(x_grid, y_grid, z_cut_count, levels=[threshold],
                            colors='red', linestyles='dashed')

    # Create legend
    legend_elements = [
        Line2D([0], [0], color='magenta', linestyle='dashed',
               label='2$\sigma$ from DNN'),
        Line2D([0], [0], color='red', linestyle='dashed',
               label='2$\sigma$ from cut and count')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Axis labels and title
    plt.xlabel("HNL mass [GeV]", fontsize=12)
    plt.ylabel("log($U^2$)", fontsize=12)
    plt.title("DNN Exclusion Plot", fontsize=14)

    # Annotate each bin with its weighted sum (for debugging or clarity)
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            if hist[i, j] > 0:
                plt.text(x_edges[i] + 0.2,
                         y_edges[j] + 0.1,
                         f"{hist[i, j]:.1f}",
                         color='white',
                         fontsize=7)

    # Save the figure and show
    plt.savefig('exclusionplot_DNN_10fb_run11.pdf', format='pdf')
    plt.show()


def main():
    # Load your JSON data
    with open('DNN_results_run6.json', 'r') as file:
        data = json.load(file)

    # Generate the plot
    make_hist_2D(data)


if __name__ == "__main__":
    main()

