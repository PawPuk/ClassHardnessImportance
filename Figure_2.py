"""This module generates datasets that were used to create the Figure 2 of the paper."""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def clean_axes():
    """Helper function that removes the axis decoration."""
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")


# ============================================================
# 1. Binary between-class imbalance
# ============================================================
def generate_binary_imbalanced(n_majority=500, n_minority=100):
    X_majority = np.random.randn(n_majority, 2) + np.array([0, 0])
    X_minority = np.random.randn(n_minority, 2) + np.array([3, 3])

    X = np.vstack([X_majority, X_minority])
    y = np.array([0]*n_majority + [1]*n_minority)

    return X, y


X_bin, y_bin = generate_binary_imbalanced()

plt.figure()
plt.scatter(X_bin[y_bin == 0, 0], X_bin[y_bin == 0, 1], alpha=0.6)
plt.scatter(X_bin[y_bin == 1, 0], X_bin[y_bin == 1, 1], alpha=0.6)
plt.title("Binary Between-Class Imbalance")
clean_axes()
plt.show()


# ============================================================
# 2. Multiclass between-class imbalance (Pareto-distributed)
# ============================================================
def generate_pareto_multiclass(n_classes=5, total_samples=500, alpha=0.66, center_scale=10):
    # Pareto weights
    raw_weights = np.random.pareto(alpha, n_classes) + 1
    print(raw_weights)
    weights = raw_weights / raw_weights.sum()
    class_sizes = (weights * total_samples).astype(int)
    class_sizes[0] += total_samples - class_sizes.sum()

    centers = np.random.uniform(low=-center_scale, high=center_scale, size=(n_classes, 2))
    X_list, y_list = [], []

    for i, (n, center) in enumerate(zip(class_sizes, centers)):
        X_i = 1.5*np.random.randn(n, 2) + center
        y_i = np.full(n, i)
        X_list.append(X_i)
        y_list.append(y_i)

    return np.vstack(X_list), np.concatenate(y_list), class_sizes

np.random.seed(66)

X_multi, y_multi, class_sizes = generate_pareto_multiclass()

plt.figure()
for cls in np.unique(y_multi):
    plt.scatter(
        X_multi[y_multi == cls, 0],
        X_multi[y_multi == cls, 1],
        alpha=0.6
    )

plt.title("Multiclass Between-Class Imbalance (Pareto)")
clean_axes()
plt.show()


# ============================================================
# 3. Binary within-class imbalance (equal class size, unequal clusters)
# ============================================================
def generate_within_class_imbalanced_equal_classes(
    samples_per_class=400
):
    # ----- Class 0 -----
    centers_0 = [(-2, -2), (-2, 14), (14, -2)]
    proportions_0 = np.array([0.6, 0.33, 0.07])
    counts_0 = (proportions_0 * samples_per_class).astype(int)
    counts_0[0] += samples_per_class - counts_0.sum()

    X0 = np.vstack([
        2*np.random.randn(n, 2) + np.array(center)
        for n, center in zip(counts_0, centers_0)
    ])
    y0 = np.zeros(samples_per_class)

    # ----- Class 1 -----
    centers_1 = [(6, 6), (14, 6), (6, 14), (16, 15)]
    proportions_1 = np.array([0.31, 0.31, 0.31, 0.07])
    counts_1 = (proportions_1 * samples_per_class).astype(int)
    counts_1[0] += samples_per_class - counts_1.sum()

    X1 = np.vstack([
        1.75*np.random.randn(n, 2) + np.array(center)
        for n, center in zip(counts_1, centers_1)
    ])
    y1 = np.ones(samples_per_class)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


X_wc, y_wc = generate_within_class_imbalanced_equal_classes()

plt.figure()
plt.scatter(X_wc[y_wc == 0, 0], X_wc[y_wc == 0, 1], alpha=0.6)
plt.scatter(X_wc[y_wc == 1, 0], X_wc[y_wc == 1, 1], alpha=0.6)
plt.title("Binary Within-Class Imbalance (Equal Class Cardinality)")
clean_axes()
plt.show()

# ============================================================
# 4. Binary: 2D circular vs 1D linear support
# ============================================================

def generate_circular_vs_linear(n_per_class=100):
    # Red class: circular (2D isotropic Gaussian)
    X_red = 0.3*np.random.randn(n_per_class, 2)
    y_red = np.zeros(n_per_class)

    # Green class: linear (1D support embedded in 2D)
    t1 = np.random.uniform(1, 1, size=n_per_class)
    t2 = np.random.uniform(-1, 1, size=n_per_class)
    noise1 = 0.03 * np.random.randn(n_per_class)
    noise2 = 0.06 * np.random.randn(n_per_class)
    X_green = np.column_stack([t1 + noise1, t2 + noise2])
    y_green = np.ones(n_per_class)

    X = np.vstack([X_green, X_red])
    y = np.concatenate([y_green, y_red])

    return X, y


X4, y4 = generate_circular_vs_linear()

plt.figure()
plt.scatter(X4[y4 == 0, 0], X4[y4 == 0, 1], color="red", alpha=0.6)
plt.scatter(X4[y4 == 1, 0], X4[y4 == 1, 1], color="green", alpha=0.6)
plt.title("Figure 4: Equal Cardinality, Different Support Dimension")
plt.legend(frameon=False)
clean_axes()
plt.show()

# ============================================================
# 5. Binary: Same geometry, different Lebesgue measure
# ============================================================

def generate_large_vs_small(n_per_class=200):
    # Green: compact cluster
    center_green = (-2, 0)
    X_green = center_green + 0.66 * np.random.randn(n_per_class, 2)
    y_green = np.zeros(n_per_class)

    # Red: same shape, larger spatial extent
    center_red = (2, 0)
    X_red = center_red + 1.1 * np.random.randn(n_per_class, 2)
    y_red = np.ones(n_per_class)

    X = np.vstack([X_green, X_red])
    y = np.concatenate([y_green, y_red])

    return X, y


X5, y5 = generate_large_vs_small()

plt.figure()
plt.scatter(X5[y5 == 0, 0], X5[y5 == 0, 1], color="green", alpha=0.6)
plt.scatter(X5[y5 == 1, 0], X5[y5 == 1, 1], color="red", alpha=0.6)
plt.title("Figure 5: Equal Cardinality, Different Spatial Scale")
clean_axes()
plt.show()

# ============================================================
# 6. Binary: Different persistent homology
# ============================================================

def generate_donut_vs_circle_left_right(n_per_class=200):
    # ----- Red class: donut (annulus), left -----
    r_inner, r_outer = 1.5, 3.0
    radius_red = np.sqrt(np.random.uniform(r_inner**2, r_outer**2, n_per_class))
    angle_red = np.random.uniform(0, 2*np.pi, n_per_class)
    X_red = np.column_stack([radius_red * np.cos(angle_red), radius_red * np.sin(angle_red)])
    X_red += 0.05 * np.random.randn(n_per_class, 2)
    X_red[:, 0] -= 4.0   # shift left
    y_red = np.ones(n_per_class)

    # ----- Green class: solid circle, right -----
    radius_green = np.sqrt(np.random.uniform(0, 9.0, n_per_class))
    angle_green = np.random.uniform(0, 2*np.pi, n_per_class)
    X_green = np.column_stack([radius_green * np.cos(angle_green), radius_green * np.sin(angle_green)])
    X_green += 0.05 * np.random.randn(n_per_class, 2)
    X_green[:, 0] += 4.0  # shift right
    y_green = np.zeros(n_per_class)

    X = np.vstack([X_green, X_red])
    y = np.concatenate([y_green, y_red])

    return X, y


X6, y6 = generate_donut_vs_circle_left_right()

plt.figure()
plt.scatter(X6[y6 == 0, 0], X6[y6 == 0, 1], color="green", alpha=0.6)
plt.scatter(X6[y6 == 1, 0], X6[y6 == 1, 1], color="red", alpha=0.6)
plt.title("Figure 6: Equal Cardinality, Topology + Spatial Shift")
clean_axes()
plt.show()
