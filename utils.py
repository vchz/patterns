import numpy as np
from itertools import product
from functools import partial
import re
from tqdm import tqdm
from collections import defaultdict

def add_superfluous_functions(X, T, basis, true_model):
    """
    Add basis functions to the true model one by one, and track score changes.

    Parameters:
        X : ndarray of shape (Nt, d)
            Trajectory data.
        T : float
            Total simulation time.
        basis : dict
            Dictionary of all available basis functions.
        true_model : dict
            Dictionary of ground-truth basis functions.

    Returns:
        log_liks : ndarray
            Log-likelihood after each added function.
        errors : ndarray
            Reconstruction after each added function.
    """
    dt = T/X.shape[0]
    all_labels = list(basis.keys())
    true_labels = list(true_model.keys())

    remaining_labels = list(set(all_labels) - set(true_labels))
    np.random.shuffle(remaining_labels)

    log_liks = []
    errors = []
    size_basis = []
    
    current_labels = true_labels

    for _label in remaining_labels:
        current_labels = current_labels + [_label]
        current_model = {label: basis[label] for label in current_labels}

        coeffs = infer_force_coefficients_einsum(X, T, current_model)
        log_lik = compute_log_likelihood(X, T, coeffs, current_model)
        error = compute_reconstruction_error(X, T, coeffs, current_model)

        log_liks.append(log_lik)
        errors.append(error)
        size_basis.append(len(current_model))

    log_liks = np.array(log_liks)
    errors = np.array(errors)
    return log_liks, errors

def evaluate_basis(X, basis_functions):
    """
    Evaluate a dictionary of vector-valued basis functions on a trajectory X.

    Parameters:
        X : ndarray of shape (Nt, d, Nx, ...)
            Input spatiotemporal field.
        basis_functions : dict
            Dictionary mapping LaTeX-style labels to basis function callables.

    Returns:
        B : ndarray of shape (Nt, nB, d, Nx, ...)
            Evaluated basis functions at each space-time point.
    """
    
    Nt, d, *Nx = X.shape
    nB = len(basis_functions)
    B = np.zeros((nB, d, Nt, *Nx))
    X = np.moveaxis(X, 0, 1)
    for i, f in enumerate(basis_functions.values()):
        B[i, :] = f(X)
    return np.moveaxis(B, 2, 0)


def greedy_basis_search(
    X, args, full_basis, sfi_engine, true_model=None, ntrials=None,
    method='PASTIS', p=0.001, max_moves=40):
    """
    Perform greedy model selection by iteratively adding/removing basis functions
    to maximize a score (e.g. PASTIS, AIC).

    Parameters:
        X : ndarray
            Observed trajectory.
        args : tuple (T, L)
            Simulation duration and domain size.
        full_basis : dict
            Dictionary of all candidate basis functions.
        sfi_engine : tuple
            Tuple of three functions: (infer_force, compute_score, compute_diffusion).
        true_model : dict or None
            If provided, the score of the true model is also evaluated.
        ntrials : int
            Number of random initializations.
        method : str
            Score function to maximize: 'PASTIS', 'AIC'.
        p : float
            Prior for PASTIS.
        max_moves : int
            Maximum number of greedy updates per trial.

    Returns:
        best_basis : dict
            Selected basis functions.
        best_coeffs : ndarray
            Corresponding inferred coefficients.
        best_score : float
            Final score.
        true_score : float or None
            Score of the true model if provided.
    """
    scores_ref = {'PASTIS': 0, 'AIC': 1}

    if method not in scores_ref:
        raise ValueError("Score method must be one of: 'PASTIS', 'AIC'")

    labels = list(full_basis.keys())
    n0 = len(labels)
    if ntrials is None:
        ntrials = n0
        
    infer_force_coefficients, compute_scores, compute_mean_diffusion = sfi_engine
    if not callable(infer_force_coefficients) or not callable(compute_scores):
        raise ValueError("SFI engine must b a tuple with two callable functions, \
                         the first to infer the parameters, and the second to compute the scores")
    
    best_overall_score = -np.inf
    best_model = set()

    # Initial models: empty, full, and random subsets
    initial_models = [set(), set(range(n0))]
    for _ in range(ntrials):
        size = np.random.randint(1, n0 + 1)
        initial_models.append(set(np.random.choice(n0, size=size, replace=False)))


    for trial, init_model in enumerate(initial_models):
        current_model = init_model
        best_score = -np.inf
        accepted_moves = 0

        with tqdm(total=max_moves, desc=f"Trial {trial+1}/{len(initial_models)}") as pbar:
            while accepted_moves < max_moves:
                neighbors = []
                for i in range(n0):
                    neighbor = current_model - {i} if i in current_model else current_model | {i}
                    if neighbor:
                        neighbors.append(neighbor)

                np.random.shuffle(neighbors)
                improved = False

                for neighbor in neighbors:
                    selected_labels = [labels[i] for i in neighbor]
                    basis_subset = {label: full_basis[label] for label in selected_labels}

                    coeffs = infer_force_coefficients(X, args, basis_subset)
                    scores = compute_scores(X, args, coeffs, basis_subset, n0, p)
                    score = scores[scores_ref[method]]

                    if score > best_score:
                        current_model = neighbor
                        best_score = score
                        accepted_moves += 1
                        pbar.update(1)
                        improved = True
                        break

                if not improved:
                    break

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_model = current_model

    final_labels = [labels[i] for i in best_model]
    final_basis = {label: full_basis[label] for label in final_labels}
    final_coeffs = infer_force_coefficients(X, args, final_basis)

    if true_model is not None:
        true_coeffs = infer_force_coefficients(X, args, true_model)
        true_scores = compute_scores(X, args, true_coeffs, true_model, n0, p)
        true_score = true_scores[scores_ref[method]]
        return final_basis, final_coeffs, best_overall_score, true_score

    return final_basis, final_coeffs, best_overall_score


def _grad(u, axis, dx):
    """
    Compute the finite-difference gradient along a spatial axis with periodic boundaries.

    Parameters:
        u : ndarray of shape (Nt, Nx, ...)
        axis : int
            Spatial axis index.
        dx : float
            Grid spacing.

    Returns:
        grad_u : ndarray of same shape as u
            Spatial gradient.
    """
    return (np.roll(u, -1, axis=axis + 1) - np.roll(u, 1, axis=axis + 1)) / (2 * dx)


def _laplacian(u, dx):
    """
    Compute the discrete Laplacian over all spatial axes of a field u.

    Parameters:
        u : ndarray of shape (Nt, Nx, ...)
        dx : float
            Grid spacing.

    Returns:
        lap_u : ndarray
            Laplacian of u.
    """
    lap = np.zeros_like(u)
    for a in range(len(u.shape[1:])):
        lap += (np.roll(u, -1, axis=a + 1) + np.roll(u, 1, axis=a + 1) - 2 * u) / dx**2
    return lap


def _poly_term(u, p, e):
    """
    Compute a polynomial basis term of the form e_i * Π_j u_j^{p_j}.

    Parameters:
        u : ndarray of shape (d, Nt, Nx, ...)
        p : tuple of int
            Exponents for each field variable.
        e : ndarray of shape (d,)
            Unit vector indicating the output dimension.

    Returns:
        result : ndarray of shape (d, Nt, Nx, ...)
    """
    prod = np.ones_like(u[0])
    for j in range(len(p)):
        prod *= u[j] ** p[j]
    return e.reshape((e.shape[0],) + (1,) * (prod.ndim)) * prod[None, ...]


def _grad_term(u, j, a, e, dx):
    """
    Compute a gradient basis term: e_i * ∂_{x_a} u_j.

    Parameters:
        u : ndarray of shape (d, Nt, Nx, ...)
        j : int
            Field index to differentiate.
        a : int
            Spatial axis.
        e : ndarray
            Unit vector for output dimension.
        dx : float
            Grid spacing.

    Returns:
        result : ndarray of shape (d, Nt, Nx, ...)
    """
    b = _grad(u[j], axis=a, dx=dx)
    return e.reshape((e.shape[0],) + (1,) * (b.ndim)) * b[None, ...]


def _lap_term(u, j, e, dx):
    """
    Compute a Laplacian basis term: e_i * Δ u_j.

    Parameters:
        u : ndarray of shape (d, Nt, Nx, ...)
        j : int
            Field index to differentiate.
        e : ndarray
            Unit vector for output dimension.
        dx : float
            Grid spacing.

    Returns:
        result : ndarray of shape (d, Nt, Nx, ...)
    """
    b = _laplacian(u[j], dx=dx)
    return e.reshape((e.shape[0],) + (1,) * (b.ndim)) * b[None, ...]


def _mixed_term(u, i, j, a, e, dx):
    """
    Compute a nonlinear gradient basis term: e_i * (u_i * ∂_{x_a} u_j).

    Parameters:
        u : ndarray of shape (d, Nt, Nx, ...)
        i, j : int
            Field indices.
        a : int
            Spatial axis.
        e : ndarray
            Unit vector for output dimension.
        dx : float
            Grid spacing.

    Returns:
        result : ndarray of shape (d, Nt, Nx, ...)
    """
    b = u[i] * _grad(u[j], axis=a, dx=dx)
    return e.reshape((e.shape[0],) + (1,) * (b.ndim)) * b[None, ...]


def get_function_from_label(label, field_dim=2, dx=None):
    """
    Parse a LaTeX-style label and return the corresponding basis function.

    Parameters:
        label : str
            Symbolic label (e.g. "$e_{0} u_{1}^2$").
        field_dim : int
            Number of fields.
        dx : float or None
            Spatial step (required for gradients and Laplacians).

    Returns:
        basis_func : callable
            Basis function as a partial.
    """
    pattern = re.compile(r'e_\{(\d+)\}\s*(.*)')
    clean_label = label.strip('$')
    match = pattern.match(clean_label)

    if not match:
        raise ValueError(f"Unrecognized label format: {label}")

    i = int(match.group(1))
    expr = match.group(2).strip()
    e_i = np.eye(field_dim)[i]

    # Constant term
    if expr == "":
        return partial(_poly_term, p=(0,) * field_dim, e=e_i)

    if dx is not None:
        # Laplacian
        if r'\Delta' in expr:
            j = int(re.search(r'u_\{(\d+)\}', expr).group(1))
            return partial(_lap_term, j=j, e=e_i, dx=dx)

        # Mixed term
        if ' ' in expr and r'\partial' in expr:
            mixed_match = re.match(r'u_\{(\d+)\} \\partial_\{x_(\d+)\} u_\{(\d+)\}', expr)
            if mixed_match:
                i_m, a, j = map(int, mixed_match.groups())
                return partial(_mixed_term, i=i_m, j=j, a=a, e=e_i, dx=dx)

        # Gradient
        grad_match = re.match(r'\\partial_\{x_(\d+)\} u_\{(\d+)\}', expr)
        if grad_match:
            a, j = map(int, grad_match.groups())
            return partial(_grad_term, j=j, a=a, e=e_i, dx=dx)

    # Polynomial
    mono_matches = re.findall(r'u_\{(\d+)\}\^?\{?(\d*)\}?', expr)
    powers = [0] * field_dim
    for j_str, p_str in mono_matches:
        j = int(j_str)
        p = int(p_str) if p_str else 1
        powers[j] = p
    return partial(_poly_term, p=tuple(powers), e=e_i)


def get_model_from_labels(latex_labels, field_dim=2, dx = None):
    """
    Convert a list of symbolic LaTeX labels into a dictionary of basis functions.

    Parameters:
        latex_labels : list of str
            Labels representing basis terms.
        field_dim : int
            Number of field components.
        dx : float or None
            Spatial step size.

    Returns:
        model_dict : dict
            Dictionary {label: basis function}
    """
    return {label: get_function_from_label(label, field_dim=field_dim, dx=dx) for label in latex_labels}


def get_basis(field_dim=2, degree=3, n = None, dx = None, nofit=[]):
    """
    Construct a dictionary of all basis functions up to a polynomial degree,
    optionally including gradients, Laplacians, and nonlinear terms.

    Parameters:
        field_dim : int
            Number of field components.
        degree : int
            Max total polynomial degree.
        n : int or None
            Number of spatial axes.
        dx : float or None
            Grid spacing for differential terms.
        nofit : list of int
            Field indices to exclude from fitting.

    Returns:
        basis_dict : dict
            Dictionary {LaTeX label: basis function}
    """
    d = field_dim
    basis = {}

    # Polynomial terms
    for i in range(d):
        if i in nofit:
            continue
        e_i = np.eye(d)[i]
        for deg in range(0, degree + 1):
            for powers in product(range(deg + 1), repeat=d):
                if sum(powers) == deg:
                    monomial = " ".join([f"u_{{{j}}}^{{{p}}}" for j, p in enumerate(powers) if p > 0])
                    label = f"e_{{{i}}} {monomial}".strip() or f"e_{{{i}}}"
                    basis[f"${label}$"] = partial(_poly_term, p=powers, e=e_i)

    if dx is not None and n is not None:
        # First derivatives and mixed terms
        for i in range(d):
            if i in nofit:
                continue
            e_i = np.eye(d)[i]
            for j in range(d):
                for a in range(n):
                    label_grad = f"e_{{{i}}} \\partial_{{x_{a}}} u_{{{j}}}"
                    basis[f"${label_grad}$"] = partial(_grad_term, j=j, a=a, e=e_i, dx=dx)
                    label_mix = f"e_{{{i}}} u_{{{j}}} \\partial_{{x_{a}}} u_{{{j}}}"
                    basis[f"${label_mix}$"] = partial(_mixed_term, i=i, j=j, a=a, e=e_i, dx=dx)

        # Laplacians
        for i in range(d):
            if i in nofit:
                continue
            e_i = np.eye(d)[i]
            for j in range(d):
                label = f"e_{{{i}}} \\Delta u_{{{j}}}"
                basis[f"${label}$"] = partial(_lap_term, j=j, e=e_i, dx=dx)

    return basis


def get_latex_model(basis_dict, coeffs, time_symbol='t', field_symbol='u', scale_factors=None):
    """
    Generate a LaTeX string representing the inferred PDE system.

    Parameters:
        basis_dict : dict
            Dictionary of selected basis functions.
        coeffs : ndarray
            Corresponding coefficients.
        time_symbol : str
            Variable name for time (default: 't').
        field_symbol : str
            Variable name for the field (default: 'u').
        scale_factors : array or None
            Optional scaling for each field component.

    Returns:
        latex_str : str
            LaTeX-formatted string representing the PDE model.
    """
    
    def format_coeff(c):
        """Format the coefficient for LaTeX, converting scientific notation when necessary."""
        if np.isclose(abs(c), 1.0):
            return "-" if c < 0 else "+"
        if abs(c) < 1e-3 or abs(c) >= 1e4:
            base = f"{c:.1e}"
            parts = base.split("e")
            mantissa = float(parts[0])
            exponent = int(parts[1])
            return f"{'-' if c < 0 else '+'}{abs(mantissa)}\\times 10^{{{exponent}}}"
        else:
            return f"{c:.5f}" if c < 0 else f"+{c:.5f}"

    assert len(basis_dict) == len(coeffs), "Mismatch between basis and coefficients"

    component_terms = defaultdict(list)
    pattern = re.compile(r'\$e_\{(\d+)\}\s*(.*)\$')

    for (label, c) in zip(basis_dict.keys(), coeffs):
        match = pattern.match(label)
        if not match or np.isclose(c, 0.0):
            continue
        i, expr = match.groups()
        i = int(i)

        scale = 1.0 if scale_factors is None else scale_factors[i]
        c_scaled = float(c) * scale

        # Format coefficient and term
        if expr.strip() == "":
            term = format_coeff(c_scaled)
        else:
            coeff_str = format_coeff(c_scaled)
            term = f"{coeff_str}{expr}"

        component_terms[i].append(term)

    if not component_terms:
        return f"\\frac{{\\partial \\mathbf{{{field_symbol}}}}}{{\\partial {time_symbol}}} = 0"

    eqns = []
    for i in sorted(component_terms.keys()):
        rhs = " ".join(component_terms[i]).lstrip('+')
        eqn = f"\\frac{{\\partial {field_symbol}_{{{i}}}}}{{\\partial {time_symbol}}} &= {rhs} \\notag"
        eqns.append(eqn)

    aligned_block = "\\begin{align}\n" + " \\\\\n".join(eqns) + "\n\\end{align}"
    return aligned_block
