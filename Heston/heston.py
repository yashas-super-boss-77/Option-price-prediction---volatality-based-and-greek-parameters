import numpy as np

def pred_options(S0, K, T, CP):
    """"
    S0 -> Stock price
    K -> strike price
    T -> Time
    CP -> Call (1) or put (-1)
    """
    # Parameters for the Heston model
    # S0 = 17080.70    # initial stock price
    r = 0.10    # risk-free interest rate
    # T = 7.0/365.0     # time to maturity
    kappa = 2.0 # mean reversion speed
    theta = 0.1 # long-term volatility level
    sigma = 0.3 # volatility of volatility
    rho = -0.5  # correlation between the stock price and its volatility
    V0 = 0.1    # initial volatility

    # Option parameters
    # K = 16600    # strike price
    CP = 1     # call option (1) or put option (-1)
    M = 10000  # number of Monte Carlo simulations

    # Simulation parameters
    dt = 0.001 # time step for the Euler-Maruyama method
    N = int(T/dt) # number of time steps

    # Generate random normal variables for the stock price and its volatility
    Z1 = np.random.normal(size=(M,N))
    Z2 = rho*Z1 + np.sqrt(1-rho**2)*np.random.normal(size=(M,N))

    # Initialize arrays for the stock price and its volatility
    S = np.zeros((M,N+1))
    V = np.zeros((M,N+1))
    S[:,0] = S0
    V[:,0] = V0

    # Perform the Monte Carlo simulation
    for i in range(1,N+1):
        V[:,i] = V[:,i-1] + kappa*(theta - V[:,i-1])*dt + sigma*np.sqrt(V[:,i-1])*np.sqrt(dt)*Z2[:,i-1]
        S[:,i] = S[:,i-1]*np.exp((r - 0.5*V[:,i-1])*dt + np.sqrt(V[:,i-1])*np.sqrt(dt)*Z1[:,i-1])

    # Compute the option price
    if CP == 1:
        payoff = np.maximum(S[:,N]-K, 0)
    else:
        payoff = np.maximum(K-S[:,N], 0)

    price = np.mean(payoff)*np.exp(-r*T)

    return price