import os  
import re  
import matplotlib.pyplot as plt

# ---- choose which folder of ranked_d{d}.txt files to plot (path relative to this script) ----
#RANKED_DIR = 'results_charact/observable_characteristics/obs_characteristics_details_d2_10'
RANKED_DIR = 'results_charact/obschar_compbasisstate/obscharacteristics_compbasis_details_d2_10'
# RANKED_DIR = 'results_charact/obscharwithmatrix_results/obscharwithmatrix_details_d2_4'

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, RANKED_DIR)  # full path to the chosen folder
# find every ranked_d{d}.txt in the folder and collect the d values, sorted
DS = sorted(int(m.group(1)) for f in os.listdir(DATA_DIR) if (m := re.fullmatch(r'ranked_d(\d+)\.txt', f)))
PLOT_DIR = os.path.join(DATA_DIR, 'plots')  # plots are saved next to the data they come from
os.makedirs(PLOT_DIR, exist_ok=True)

def parse_ranked(path):
    """Returns (rhs, samples) from a ranked_d{d}.txt, where samples is a list of
    (lhs, A1 eigvals, A2 eigvals). B1, B2 equal A1, A2 so they are skipped."""
    text = open(path, 'r').read()  # read the whole file as one string
    rhs = float(re.search(r'rhs = (\S+)', text).group(1))  # the bound 2-4/(2+d), written once in the header
    samples = []  # will hold one (lhs, A1 eigvals, A2 eigvals) tuple per sample
    for block in text.split('=' * 60)[1:]:  # samples are separated by a line of 60 '='; [1:] drops the header
        lhs = float(re.search(r'lhs = (\S+)', block).group(1))  # this sample's |Cor{A1,B1}|+|Cor{A2,B2}|
        eigs = {}  # maps 'A1' / 'A2' to its list of eigenvalues
        # find "A1: trace norm ... eigvals = [ ... ]" and the same for A2; DOTALL lets .*? cross line breaks
        for name, vals in re.findall(r'(A1|A2): trace norm.*?eigvals = \[(.*?)\]', block, re.DOTALL):
            eigs[name] = [float(v) for v in vals.split()]  # split the bracket contents on spaces into floats
        samples.append((lhs, eigs['A1'], eigs['A2']))  # store this sample
    return rhs, samples  # give back the bound and all parsed samples


for d in DS:  # make a separate figure for each d
    rhs, samples = parse_ranked(os.path.join(DATA_DIR, f'ranked_d{d}.txt'))  # load this d's data
    plt.figure(figsize=(6, 4))  # start a new, separate figure window
    for lhs, a1, a2 in samples:  # go through every sample
        plt.scatter(a1, [lhs] * len(a1), c='g', s=10, label='A1')  # each A1 eigenvalue as a point at height lhs
        plt.scatter(a2, [lhs] * len(a2), c='b', s=10, label='A2')  # same for A2, in blue
    plt.axhline(rhs, c='r', ls='--', label='rhs')  # dashed red horizontal line at the bound
    plt.title(f"lhs vs eigenvals, d = {d}")  # title shows which d this figure is
    plt.xlabel('eigenvals')  # x axis: eigenvalue
    plt.ylabel('lhs')  # y axis: lhs of the sample the eigenvalue belongs to
    handles, labels = plt.gca().get_legend_handles_labels()  # every scatter call added a legend entry
    unique = dict(zip(labels, handles))  # keep only one entry per label (A1, A2, rhs)
    plt.legend(unique.values(), unique.keys())  # draw the legend without duplicates
    plt.tight_layout()  # stop labels from being cut off
    plt.savefig(os.path.join(PLOT_DIR, f'obs_eigenvals_d{d}.png'), dpi=300, bbox_inches='tight')  # save this figure
plt.show()  # open all the figures at once
