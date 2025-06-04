import numpy as np


def cosine_similarity_numpy(X, Y=None, *, dense_output=True):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got array with shape {X.shape}")
    _, n_features = X.shape

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D; got array with shape {Y.shape}")
        if Y.shape[1] != n_features:
            raise ValueError(
                f"Number of features of X and Y must match; "
                f"got X.shape[1]={n_features}, Y.shape[1]={Y.shape[1]}"
            )
            
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)

    X_norm_safe = X_norm.copy()
    Y_norm_safe = Y_norm.copy()
    X_norm_safe[X_norm_safe == 0] = 1.0
    Y_norm_safe[Y_norm_safe == 0] = 1.0
    
    dot_products = X.dot(Y.T)
    
    norm_matrix = np.outer(X_norm_safe, Y_norm_safe)
    
    S = dot_products / norm_matrix

    return S


class StandardScalerNumpy:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        # These will be set in fit()
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features).")

        if self.with_mean:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=float)

        if self.with_std:
            
            var = ((X - self.mean_) ** 2).mean(axis=0)
            
            self.scale_ = np.sqrt(var)
            
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1], dtype=float)

        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScalerNumpy instance is not fitted yet. "
                             "Call 'fit' with appropriate data before using 'transform'.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features).")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"Number of features of the input ({X.shape[1]}) "
                             f"does not match the fitted data ({self.mean_.shape[0]}).")

        # center
        X_centered = X - self.mean_
        # scale
        X_scaled = X_centered / self.scale_
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScalerNumpy instance is not fitted yet.")
        X_scaled = np.asarray(X_scaled, dtype=float)
        if X_scaled.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features).")
        if X_scaled.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"Number of features of the input ({X_scaled.shape[1]}) "
                             f"does not match the fitted data ({self.mean_.shape[0]}).")

        return X_scaled * self.scale_ + self.mean_


class PCANumpy:
    def __init__(self, n_components=None):
        self.n_components = n_components
        # These will be populated in fit()
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features).")
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # Determine how many components to keep
        max_possible = min(n_samples, n_features)
        if self.n_components is None:
            k = max_possible
        else:
            k = self.n_components
        self.n_components_ = k
        
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.singular_values_ = S[:k]                  # shape: (k,)
        self.components_ = Vt[:k, :]                   # shape: (k, n_features)
        
        explained_var = (self.singular_values_ ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_var       # shape: (k,)
        
        total_var = ((X_centered ** 2).sum(axis=0) / (n_samples - 1)).sum()
        
        self.explained_variance_ratio_ = explained_var / total_var  # shape: (k,)

        return self

    def transform(self, X):
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCANumpy instance is not fitted yet. Call 'fit' first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features).")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) does not match fitted data ({self.n_features_})."
            )

        X_centered = X - self.mean_
        
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCANumpy instance is not fitted yet. Call 'fit' first.")
        X_t = np.asarray(X_transformed, dtype=float)
        if X_t.ndim != 2:
            raise ValueError("Input must be 2D (n_samples, n_components_).")
        if X_t.shape[1] != self.n_components_:
            raise ValueError(
                f"Number of components in X_transformed ({X_t.shape[1]}) "
                f"does not match fitted n_components_ ({self.n_components_})."
            )
            
        return np.dot(X_t, self.components_) + self.mean_


class GaussianMixtureNumpy:
    def __init__(
        self,
        n_components,
        tol=1e-3,
        max_iter=100,
        reg_covar=1e-6,
        random_state=None
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state
        
        # Attributes to be learned in fit:
        self.weights_ = None        # shape (n_components,)
        self.means_ = None          # shape (n_components, n_features)
        self.covariances_ = None    # shape (n_components, n_features, n_features)
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf
    
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # 1 Weights: uniform
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        
        # 2 Means: pick K distinct samples
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        
        # 3 Covariances: initialize each to the data covariance + reg_covar * I
        X_centered = X - X.mean(axis=0)
        empirical_covar = (X_centered.T @ X_centered) / n_samples
        self.covariances_ = np.array([
            empirical_covar + self.reg_covar * np.eye(n_features)
            for _ in range(self.n_components)
        ])
    
    def _estimate_log_gaussian_prob(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        const_term = n_features * np.log(2 * np.pi)
        
        for k in range(self.n_components):
            mean_k = self.means_[k]
            covar_k = self.covariances_[k]
            
            # Compute Cholesky for stability: covar_k = L L^T
            try:
                L = np.linalg.cholesky(covar_k)
            except np.linalg.LinAlgError:
                # If covariance is not PD (should not happen if reg_covar > 0),
                # we add more regularization and retry.
                covar_k += self.reg_covar * np.eye(n_features)
                L = np.linalg.cholesky(covar_k)
            
            # Solve (X - mean) efficiently via L
            X_centered = X - mean_k  # shape (n_samples, n_features)
            
            # Solve L * y = X_centered^T  => y = solve(L, X_centered^T)
            # Then compute squared Mahalanobis: sum(y^2) row‐wise
            y = np.linalg.solve(L, X_centered.T)  # shape (n_features, n_samples)
            mahala_sq = np.sum(y ** 2, axis=0)   # shape (n_samples,)
            
            # log determinant = 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            
            # log N(x) = -0.5 * (const_term + log_det + mahalanobis_sq)
            log_prob[:, k] = -0.5 * (const_term + log_det + mahala_sq)
        
        return log_prob  # shape (n_samples, n_components)
    
    def _estimate_log_weights(self):
        return np.log(self.weights_)
    
    def _estimate_log_prob_resp(self, X):
        # Log Gaussian probabilities: shape (n_samples, n_components)
        log_gauss = self._estimate_log_gaussian_prob(X)
        
        # Add log weights: shape (n_samples, n_components)
        log_weights = self._estimate_log_weights()[np.newaxis, :]  # shape (1, n_components)
        weighted_log_prob = log_gauss + log_weights
        
        # For each sample i, compute log-sum-exp over k for normalization
        max_log_prob = np.max(weighted_log_prob, axis=1, keepdims=True)  # shape (n_samples, 1)
        stable_sum = np.exp(weighted_log_prob - max_log_prob)
        sum_exp = np.sum(stable_sum, axis=1, keepdims=True)  # shape (n_samples, 1)
        log_prob_norm = (max_log_prob + np.log(sum_exp)).ravel()  # shape (n_samples,)
        
        # 4) log responsibilities = weighted_log_prob - log_prob_norm[:, None]
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_prob_norm, log_resp
    
    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        
        # Effective number of samples assigned to each component k
        nk = resp.sum(axis=0)  # shape (n_components,)
        
        # Update weights
        self.weights_ = nk / n_samples
        
        # Update means
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]  # shape (n_components, n_features)
        
        # Update covariances
        covariances = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            X_centered = X - self.means_[k]  # shape (n_samples, n_features)
            # weight each sample by resp[i, k]
            wk = resp[:, k][:, np.newaxis]  # shape (n_samples, 1)
            cov_k = (X_centered * wk).T @ X_centered  # shape (n_features, n_features)
            cov_k /= nk[k]
            # add reg_covar on diagonal
            cov_k.flat[:: n_features + 1] += self.reg_covar
            covariances[k] = cov_k
        
        self.covariances_ = covariances
    
    def _compute_lower_bound(self, log_prob_norm):
        return np.mean(log_prob_norm)
    
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features).")
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        prev_lower_bound = -np.inf
        for n_iter in range(1, self.max_iter + 1):
            # E‐step: compute log responsibilities
            log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
            
            # Convert log_resp to responsibilities
            resp = np.exp(log_resp)  # shape (n_samples, n_components)
            
            # M‐step: update parameters
            self._m_step(X, resp)
            
            # Compute average log‐likelihood (lower bound)
            curr_lower_bound = self._compute_lower_bound(log_prob_norm)
            change = curr_lower_bound - prev_lower_bound
            
            if abs(change) < self.tol:
                self.converged_ = True
                self.n_iter_ = n_iter
                self.lower_bound_ = curr_lower_bound
                break
            
            prev_lower_bound = curr_lower_bound
        
        else:
            # Did not converge within max_iter
            self.n_iter_ = self.max_iter
            self.lower_bound_ = curr_lower_bound
        
        return self
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)
    
    def predict(self, X):
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)