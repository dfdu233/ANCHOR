from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def subset_index_aux(scores: List[float], lambda_: float) -> List[int]:
    """
    Function to find the indices of the smallest subset of scores whose sum is greater than lambda_;
    these indices are returned as a list. If no such subset exists, returns all indices.
    """
    # Sort scores with their original indices in descending order
    indexed_sorted_scores = sorted(enumerate(scores), key=lambda x: -x[1])

    current_sum = 0.0
    selected_indices = []
    for i, score in indexed_sorted_scores:
        current_sum += score
        selected_indices.append(i)
        if current_sum > lambda_:
            return sorted(selected_indices)
        if score <= 0:
            # If the score is non-positive, we can stop since current_sum will not increase
            break

    return list(range(len(scores)))


def plot_admissibility_curve(ax: Axes,
                             admissibility_results: Dict[float, float], 
                             is_lambda_inf: Dict[float, bool],  
                             is_calibration: bool = False,
                             dataset_size: int | None = None
                            ) -> Axes:
    """
    Function for plotting the average admissibility values versus the list of gamma values
    Input arguments:
    - ax: the matplotlib Axes object to plot on
    - admissibility_values: a dictionary of average admissibility values, one for each gamma. Each value
                            should already be the average of the admissibility results from the dataset
                            under the current lambda threshold calibrated based on the gamma value.
    - is_lambda_inf: a dictionary of boolean values indicating whether lambda is infinity for each gamma value.
    - is_calibration (optional): default value is false for testing set, for calibration set sanity check, set
                    it to true, the slope would be adjusted accordingly
    - dataset_size (optional): required if "is_calibration" is True, which is the calibration dataset size, required 
                    to compute the adjusted slope
    """
    if len(admissibility_results) != len(is_lambda_inf):
        raise ValueError("Having different number of average admissibility values and boolean values on whether lambda is inf or not: " + \
                         f"{len(admissibility_results)} VS {len(is_lambda_inf)}")

    if is_calibration and (dataset_size is None or dataset_size <= 0):
        raise ValueError("Valid dataset_size must be provided when is_calibration is True.")
    
    for gamma_value, admissibility_res in admissibility_results.items():
        if gamma_value not in is_lambda_inf:
            raise KeyError(f"Gamma value {gamma_value} missing in is_lambda_inf dictionary.")
        
        if is_lambda_inf[gamma_value]:
            ax.scatter(x=gamma_value, 
                       y=admissibility_res, 
                       marker='x', 
                       color='red', 
                       s=70)
        else:
            ax.scatter(x=gamma_value, 
                       y=admissibility_res, 
                       marker='o', 
                       color='blue', 
                       s=60)

    # Connecting the dots
    sorted_gammas = sorted(admissibility_results.keys())
    sorted_values = [admissibility_results[g] for g in sorted_gammas]
    ax.plot(sorted_gammas, sorted_values, color='black', 
            linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)      
    
    slope_gamma = 1.0 if not is_calibration \
                        else (dataset_size+1)/dataset_size
    
    ax.axline(xy1=(0,0), slope=slope_gamma, color='gray', linestyle='--', label='Lower Bound')
    ax.set_xlabel(r'$\gamma$', fontsize=28)
    ax.set_ylabel('Mean Test Admissibility' if not is_calibration 
                    else 'Mean Calibration Admissibility', fontsize=28)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.grid(True, color='gray', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # legends for scatter points
    ax.scatter([], [], marker="o", color="blue", label=r"$\hat{\lambda} < \infty$", s=60)
    ax.scatter([], [], marker="x", color="red", label=r"$\hat{\lambda} = \infty$", s=70)
    ax.legend(fontsize=21)

    return ax


def plot_generation_sequence_length(ax: Axes,
                                    avg_length_conformal: Dict[float, float],
                                    y_label : str = 'Mean Sequence Length',
                                    plot_diagonal : bool = False
                                    ) -> Axes:
    """ 
    Function for plotting the average number of questions as bar graph
    Input arguments:
        - ax: the matplotlib Axes object to plot on
        - avg_length_conformal: a dictionary of average sequence length for each gamma
    """
    # sort the keys for consistent deterministic ordering
    gamma_sorted = sorted(avg_length_conformal.keys())
    val_sorted = [avg_length_conformal[gamma] for gamma in gamma_sorted]

    # Do not plot the error bar for now
    ax.bar(x=gamma_sorted, 
           height=val_sorted, 
           width=0.03)

    if plot_diagonal:
        ax.axline(xy1=(0, 0), slope=1.0, color='gray', linestyle='--')
        ax.set_ylim(0, 1.03)
 
    ax.set_xlabel(r'$\gamma$', fontsize=28) 
    ax.set_ylabel(y_label, fontsize=28)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    return ax