import numpy as np

def apply_kalman_filter(signal: np.ndarray, q: float = 1e-6, r:  float = 0.09, p: float = 1.0)-> np.ndarray:
    """
    Applies a simple Kalman filter to a 1D signal.

    Args:
        signal (np.ndarray): The input noisy signal.
        q (float): Process noise covariance.
        r (float): Measurement noise covariance.
        p (float): Initial estimation error covariance.

    Returns:
        np.ndarray: The filtered signal.
    """
    x = 0.0             # Initial estimate
    P = p               # Initial estimation error
    filtered = []

    for z in signal:
        # Prediction step
        x_pred = x
        P_pred = P + q

        # Update step
        K = P_pred / (P_pred + r)           # Kalman gain
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

        filtered.append(x)

    return np.array(filtered)

def test_kalman():
    import matplotlib.pyplot as plt
    t = np.linspace(0, 2 * np.pi, 1000)
    clean = np.sin(t)
    noise = np.random.normal(0, 0.3, size = t.shape)
    noisy = clean + noise

    filtered = apply_kalman_filter(noisy)

    plt.figure(figsize=(10,5))
    plt.plot(t, noisy, label = "Noisy Signal", alpha = 0.5)
    plt.plot(t, filtered, label = "Kalman Filtered", linewidth = 2)
    plt.plot(t, clean, label = "True Signal", linestyle = "--")
    plt.legend()
    plt.title("Kalman Filter Demo")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

test_kalman()
