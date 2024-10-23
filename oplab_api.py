import os
import requests
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import os
import requests
import pandas as pd
from datetime import datetime

# NOT USED
## 1. Data Fetching and parsing functions. 
def fetch_interest(src='cetip'):
    # Keeping this function as is, assuming it fetches the appropriate risk-free rate for Brazil
    url = 'https://api.oplab.com.br/v3/market/interest_rates'
    headers = {
        'Access-Token': os.getenv('OPLAB_API_KEY')    
    }
    response = requests.get(url, headers=headers)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    df['updated_at'] = pd.to_datetime(df['updated_at']).dt.strftime('%Y-%m-%d')
    if src == 'cetip':
        return df['value'][1]/100
    else:
        return df['value'][0]/100

# USED ON 
#1.1  Fetch active option chani for a given day
def fetch_options_data():
    url = 'https://api.oplab.com.br/v3/market/options/IBOV'
    headers = {
        'Access-Token': os.getenv('OPLAB_API_KEY')    
    }

    response = requests.get(url, headers=headers)
    return parse_options_to_dataframe(response.text)


#GET HISTORICAL OPTION CHAIN  - USED 
def get_historical_options(spot, start, end, symbol=None):
    """
    Fetch historical options data from the API.

    Args:
        spot (str): The spot symbol (e.g., 'PETR4').
        start (str): The start date in 'YYYY-MM-DD' format.
        end (str): The end date in 'YYYY-MM-DD' format.
        symbol (str, optional): The option symbol (e.g., 'PETRA230'). Default is None.
    
    Returns:
        list: A list of historical options data if successful, otherwise None.
    """
    # Fetch the access token from environment variable
    access_token = os.getenv('OPLAB_API_KEY')
    
    # Check if the access token is set
    if not access_token:
        print("Error: Access token not found. Please set the 'OPLAB_API_KEY' environment variable.")
        return None

    # Construct the base URL
    url = f'https://api.oplab.com.br/v3/market/historical/options/{spot}/{start}/{end}'
    
    # Append the symbol to the URL if provided
    if symbol:
        url += f'?symbol={symbol}'

    headers = {
        'Access-Token': access_token
    }
    
    try:
        # Make the GET request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        
        # Parse the JSON response
        data = response.json()
        hist = parse_historical_options_data(data)
        hist['time'] = pd.to_datetime(hist['time'])
        hist['date'] = hist['time'].dt.normalize()
        return hist

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# used
def parse_options_to_dataframe(json_data):
    # Keeping this function as is
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    df = pd.DataFrame(data)
    
    columns = ['symbol'] + [col for col in df.columns if col != 'symbol']
    df = df[columns]
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# used
def parse_historical_options_data(data):
    """
    Parse the historical options data response into a DataFrame.

    Args:
        data (list): The list of historical options data as returned from the API.

    Returns:
        pd.DataFrame: A DataFrame containing parsed historical options data.
    """
    # List to store parsed records
    parsed_data = []

    # Iterate through each record in the response data
    for record in data:
        # Initialize a dictionary for storing flattened data
        parsed_record = {}

        # Dynamically extract fields from the main record
        for key, value in record.items():
            # If the key is 'spot', extract the 'price' value
            if key == 'spot' and isinstance(value, dict):
                parsed_record['spot_price'] = value.get('price')
            # If the key is 'time', parse it to 'YYYY-MM-DD' format
            elif key == 'time' and isinstance(value, str):
                try:
                    # Parse the time string and format it as 'YYYY-MM-DD'
                    parsed_record['time'] = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")
                except ValueError:
                    # Handle the case where the time format might be different
                    parsed_record['time'] = value  # Keep the original if parsing fails
            # Otherwise, directly add the key-value pair
            else:
                parsed_record[key] = value

        # Append the parsed record to the list
        parsed_data.append(parsed_record)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(parsed_data)
    return df

# USED
def filter_options_data(options_data):
    # Remove options with both zero bid and ask price
    filtered_data = options_data[(options_data['bid'] > 0) | (options_data['ask'] > 0)]
    
    # Allow bid == ask, but remove cases where ask < bid
    filtered_data = filtered_data[filtered_data['bid'] <= filtered_data['ask']]
    
    # Remove extreme outliers in strike prices
    median_strike = filtered_data['strike'].median()
    filtered_data = filtered_data[
        (filtered_data['strike'] >= 0.1 * median_strike) & 
        (filtered_data['strike'] <= 10 * median_strike)
    ]
    
    # Ensure we have both call and put options
    if 'category' in filtered_data.columns:
        call_options = filtered_data[filtered_data['category'] == 'CALL']
        put_options = filtered_data[filtered_data['category'] == 'PUT']
        if call_options.empty or put_options.empty:
            raise ValueError("Missing either call or put options after filtering")
    else:
        raise ValueError("'category' column not found in options data")
    
    return filtered_data

## CALCULATIONS FOR LIVE
def select_near_and_next_term(options_data):
    sorted_options = options_data.sort_values('days_to_maturity')
    unique_expirations = sorted_options['days_to_maturity'].unique()
    
    if len(unique_expirations) < 2:
        raise ValueError(f"Insufficient number of unique expirations. Available expirations: {unique_expirations}")
    
    # Select the two nearest expiration dates
    near_term = unique_expirations[0]
    next_term = unique_expirations[1]
    
    near_term_options = sorted_options[sorted_options['days_to_maturity'] == near_term]
    next_term_options = sorted_options[sorted_options['days_to_maturity'] == next_term]
    
    print(f"Near-term expiry: {near_term} days")
    print(f"Next-term expiry: {next_term} days")
    
    return near_term_options, next_term_options


def calculate_forward_price(options_data, risk_free_rate, time_to_expiration):
    call_options = options_data[options_data['category'] == 'CALL']
    put_options = options_data[options_data['category'] == 'PUT']
    
    print(f"Number of call options: {len(call_options)}")
    print(f"Number of put options: {len(put_options)}")
    
    if call_options.empty and put_options.empty:
        raise ValueError("No options available for forward price calculation")
    
    if call_options.empty or put_options.empty:
        print("Warning: Only one type of option available. Using spot price as forward price.")
        return options_data['spot_price'].iloc[0]
    
    merged_options = pd.merge(call_options, put_options, on='strike', suffixes=('_call', '_put'))
    print(f"Number of merged options: {len(merged_options)}")
    
    if merged_options.empty:
        print("Warning: No matching strike prices for call and put options. Using spot price as forward price.")
        return options_data['spot_price'].iloc[0]
    
    merged_options['price_diff'] = abs(merged_options['close_call'] - merged_options['close_put'])
    min_diff_strike = merged_options.loc[merged_options['price_diff'].idxmin(), 'strike']
    
    call_price = merged_options.loc[merged_options['strike'] == min_diff_strike, 'close_call'].values[0]
    put_price = merged_options.loc[merged_options['strike'] == min_diff_strike, 'close_put'].values[0]
    
    forward_price = min_diff_strike + np.exp(risk_free_rate * time_to_expiration) * (call_price - put_price)
    
    return forward_price


def select_strikes(options_data, forward_price):
    # Find K0 (the strike price immediately below the forward index level)
    K0 = options_data[options_data['strike'] <= forward_price]['strike'].max()
    
    # Select out-of-the-money puts
    otm_puts = options_data[(options_data['strike'] < K0) & (options_data['category'] == 'PUT')]
    
    # Select out-of-the-money calls
    otm_calls = options_data[(options_data['strike'] > K0) & (options_data['category'] == 'CALL')]
    
    # Include K0 put and call
    k0_options = options_data[options_data['strike'] == K0]
    
    return pd.concat([otm_puts, k0_options, otm_calls])

def calculate_variance(options_data, forward_price, K0, T, R):
    options_data = options_data.sort_values('strike')
    options_data['delta_K'] = options_data['strike'].diff().fillna(options_data['strike'].diff().iloc[-1])
    
    def option_contribution(row):
        Q = (row['bid'] + row['ask']) / 2
        return (row['delta_K'] / row['strike']**2) * np.exp(R * T) * Q
    
    options_data['contribution'] = options_data.apply(option_contribution, axis=1)
    
    sum_contribution = options_data['contribution'].sum()
    print(f"Sum of contributions: {sum_contribution}")
    print(f"Forward price: {forward_price}, K0: {K0}")
    
    if sum_contribution == 0 or np.isnan(sum_contribution) or np.isnan(K0):
        print("Warning: Invalid data for variance calculation. Using a simplified variance estimation.")
        # Use the average of bid-ask spread as a simple volatility estimator
        avg_spread = (options_data['ask'] - options_data['bid']).mean() / options_data['strike'].mean()
        return (avg_spread ** 2) * (365 / T)  # Annualized variance
    
    variance = (2/T) * sum_contribution - (1/T) * ((forward_price/K0 - 1)**2)
    
    if np.isnan(variance) or variance < 0:
        print(f"Warning: Calculated variance is {variance}. Using absolute value.")
        variance = abs(variance)
    
    return variance


#calculate live 
def calculate_vix():

    options_data = fetch_options_data()
    print(f"Original data shape: {options_data.shape}")
    
    filtered_options = filter_options_data(options_data)
    print(f"Filtered data shape: {filtered_options.shape}")
    
    if filtered_options.empty:
        raise ValueError("No options data available after filtering")
    
    try:
        near_term_options, next_term_options = select_near_and_next_term(filtered_options)
        print(f"Near-term options shape: {near_term_options.shape}")
        print(f"Next-term options shape: {next_term_options.shape}")
        
        risk_free_rate = fetch_interest()  # Consider fetching this dynamically
        
        T1 = near_term_options['days_to_maturity'].iloc[0] / 365
        T2 = next_term_options['days_to_maturity'].iloc[0] / 365
        
        print(f"T1: {T1}, T2: {T2}")
        
        print("Calculating forward price for near-term options:")
        forward_price_1 = calculate_forward_price(near_term_options, risk_free_rate, T1)
        print("Calculating forward price for next-term options:")
        forward_price_2 = calculate_forward_price(next_term_options, risk_free_rate, T2)
        
        print(f"Forward price 1: {forward_price_1}, Forward price 2: {forward_price_2}")
        
        K0_1 = near_term_options[near_term_options['strike'] <= forward_price_1]['strike'].max()
        K0_2 = next_term_options[next_term_options['strike'] <= forward_price_2]['strike'].max()
        
        if np.isnan(K0_1) or np.isnan(K0_2):
            print("Warning: Unable to determine K0. Using forward price as K0.")
            K0_1 = K0_1 if not np.isnan(K0_1) else forward_price_1
            K0_2 = K0_2 if not np.isnan(K0_2) else forward_price_2
        
        print("Calculating variance for near-term options:")
        variance_1 = calculate_variance(near_term_options, forward_price_1, K0_1, T1, risk_free_rate)
        print("Calculating variance for next-term options:")
        variance_2 = calculate_variance(next_term_options, forward_price_2, K0_2, T2, risk_free_rate)
        
        print(f"Variance 1: {variance_1}, Variance 2: {variance_2}")
        
        w1 = (T2 - 30/365) / (T2 - T1)
        w2 = (30/365 - T1) / (T2 - T1)
        
        print(f"Weight 1: {w1}, Weight 2: {w2}")
        
        variance_30_day = w1 * variance_1 * (T1 / (30/365)) + w2 * variance_2 * (T2 / (30/365))
        
        vix = 100 * np.sqrt(variance_30_day * 365 / 30)
        
        return vix
    except Exception as e:
        print(f"Error in VIX calculation: {str(e)}")
        print("Near-term options:")
        print(near_term_options)
        print("Next-term options:")
        print(next_term_options)
        raise


## Calculate alternate VIX for live

def estimate_implied_volatility(option, spot_price, T, risk_free_rate):
    strike = option['strike']
    option_price = option['close']
    option_type = option['category'].lower()
    
    # Simple volatility estimation based on Brenner-Subrahmanyam approximation
    if option_type == 'call':
        moneyness = np.log(spot_price / strike)
    else:  # put
        moneyness = np.log(strike / spot_price)
    
    implied_vol = np.sqrt(2 * np.pi / T) * (option_price / spot_price) * np.exp(risk_free_rate * T / 2)
    
    return min(max(implied_vol, 0.01), 2.0)  # Cap between 1% and 200%

def calculate_variance_from_prices(options_data, T, risk_free_rate):
    spot_price = options_data['spot_price'].iloc[0]
    forward_price = spot_price * np.exp(risk_free_rate * T)
    
    # Filter options to use only those within 5% of the forward price
    atm_options = options_data[
        (options_data['strike'] >= 0.95 * forward_price) & 
        (options_data['strike'] <= 1.05 * forward_price)
    ]
    
    if atm_options.empty:
        raise ValueError("No near-the-money options available for variance calculation")
    
    # Sort options by strike price
    atm_options = atm_options.sort_values('strike')
    
    # Calculate delta_K
    atm_options['delta_K'] = atm_options['strike'].diff()
    atm_options.loc[atm_options.index[0], 'delta_K'] = atm_options['strike'].iloc[1] - atm_options['strike'].iloc[0]
    atm_options.loc[atm_options.index[-1], 'delta_K'] = atm_options['strike'].iloc[-1] - atm_options['strike'].iloc[-2]
    
    # Calculate implied volatilities
    atm_options['IV'] = atm_options.apply(
        lambda row: estimate_implied_volatility(row, spot_price, T, risk_free_rate), 
        axis=1
    )
    
    # Remove outliers (IVs more than 2 standard deviations from the mean)
    mean_iv = atm_options['IV'].mean()
    std_iv = atm_options['IV'].std()
    atm_options = atm_options[(atm_options['IV'] > mean_iv - 2*std_iv) & (atm_options['IV'] < mean_iv + 2*std_iv)]
    
    # Use weighted average of implied variances
    total_weight = atm_options['delta_K'].sum()
    weighted_variance = ((atm_options['IV'] ** 2) * atm_options['delta_K']).sum() / total_weight
    
    return weighted_variance


def calculate_alternative_vix(risk_free_rate=0.11):
    options_data = fetch_options_data()
    print(f"Original data shape: {options_data.shape}")
    print(f"Using risk-free rate: {risk_free_rate}")
    
    # Filter out options with zero or NaN closing prices and add liquidity filter
    filtered_options = options_data[
        (options_data['close'] > 0) & 
        (options_data['close'].notna()) &
        (options_data['volume'] > 0)  # Basic liquidity filter
    ]
    print(f"Filtered data shape: {filtered_options.shape}")
    
    if filtered_options.empty:
        raise ValueError("No valid options data available after filtering")
    
    # Select near-term and next-term options
    sorted_options = filtered_options.sort_values('days_to_maturity')
    unique_expirations = sorted_options['days_to_maturity'].unique()
    
    if len(unique_expirations) < 2:
        raise ValueError(f"Insufficient number of unique expirations. Available expirations: {unique_expirations}")
    
    near_term = unique_expirations[0]
    next_term = unique_expirations[1]
    
    near_term_options = sorted_options[sorted_options['days_to_maturity'] == near_term]
    next_term_options = sorted_options[sorted_options['days_to_maturity'] == next_term]
    
    print(f"Near-term expiry: {near_term} days, options shape: {near_term_options.shape}")
    print(f"Next-term expiry: {next_term} days, options shape: {next_term_options.shape}")
    
    # Calculate variance for each term
    T1 = near_term / 365
    T2 = next_term / 365
    
    print("Calculating variance for near-term options:")
    variance_1 = calculate_variance_from_prices(near_term_options, T1, risk_free_rate)
    print("Calculating variance for next-term options:")
    variance_2 = calculate_variance_from_prices(next_term_options, T2, risk_free_rate)
    
    print(f"Variance 1: {variance_1}, Variance 2: {variance_2}")
    
    # Calculate weights
    w1 = (T2 - 30/365) / (T2 - T1)
    w2 = (30/365 - T1) / (T2 - T1)
    
    print(f"Weight 1: {w1}, Weight 2: {w2}")
    
    # Calculate 30-day variance
    variance_30_day = w1 * variance_1 * (T1 / (30/365)) + w2 * variance_2 * (T2 / (30/365))
    
    # Calculate VIX
    vix = 100 * np.sqrt(variance_30_day * 365 / 30)
    
    return vix

##########HISTORICAL  

def calculate_variance(options, T, r, spot_price):
    """
    Calculates the variance for a given set of options, time to expiration T, and risk-free rate r.
    """
    # Separate calls and puts
    calls = options[options['type'] == 'CALL']
    puts = options[options['type'] == 'PUT']
    
    # Merge calls and puts on strike price
    options_merged = pd.merge(calls, puts, on='strike', suffixes=('_call', '_put'))
    
    if options_merged.empty:
        return None  # Not enough data to calculate variance
    
    # Calculate F using the spot price
    F = spot_price * np.exp(r * T)
    
    # Set K0 as the strike price equal to or immediately below F
    strikes = np.sort(options['strike'].unique())
    K0 = strikes[strikes <= F].max()
    
    # Calculate Delta K
    delta_K = {}
    strikes_sorted = sorted(strikes)
    for i, K in enumerate(strikes_sorted):
        if i == 0:
            delta_K[K] = strikes_sorted[i+1] - K
        elif i == len(strikes_sorted) -1:
            delta_K[K] = K - strikes_sorted[i-1]
        else:
            delta_K[K] = (strikes_sorted[i+1] - strikes_sorted[i-1]) / 2
    
    # Sum over all strikes
    sigma_squared = 0
    for K in strikes_sorted:
        if K < K0:
            # Use OTM puts
            option_row = options[(options['strike'] == K) & (options['type'] == 'PUT')]
            if not option_row.empty:
                Q_K = option_row['premium'].iloc[0]
            else:
                continue
        elif K > K0:
            # Use OTM calls
            option_row = options[(options['strike'] == K) & (options['type'] == 'CALL')]
            if not option_row.empty:
                Q_K = option_row['premium'].iloc[0]
            else:
                continue
        else:
            # K == K0
            option_call = options[(options['strike'] == K) & (options['type'] == 'CALL')]
            option_put = options[(options['strike'] == K) & (options['type'] == 'PUT')]
            if not option_call.empty and not option_put.empty:
                Q_K_call = option_call['premium'].iloc[0]
                Q_K_put = option_put['premium'].iloc[0]
                Q_K = (Q_K_call + Q_K_put) / 2
            else:
                continue
        # Delta K
        deltaK = delta_K[K]
        # Adjust Q_K if necessary to match the scale of K^2
        Q_K_adjusted = Q_K  # Apply any necessary scaling here
        # Contribution to sigma_squared
        sigma_squared += (deltaK / (K**2)) * Q_K_adjusted * np.exp(r * T)
    # Final sigma_squared calculation
    sigma_squared = (2 / T) * sigma_squared - (1 / T) * ((F / K0 - 1) ** 2)
    print(f"F: {F}, K0: {K0}")
    print(f"deltaK: {deltaK}")
    print(f"Q_K at K={K}: {Q_K_adjusted}")
    print(f"Contribution to sigma_squared at K={K}: {(deltaK / (K**2)) * Q_K_adjusted * np.exp(r * T)}")

    return sigma_squared



## THIS ONE
def calculate_vix_df(spot, start, end):
    # Step 1: Load the data
    df = get_historical_options(spot=spot, start=start, end=end)
    if df is None or df.empty:
        print("No data returned from get_historical_options.")
        return None

    # Ensure 'time' and 'due_date' are datetime and tz-naive
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df['due_date'] = pd.to_datetime(df['due_date']).dt.tz_localize(None)

    # Convert to numeric and drop invalid rows
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['premium'] = pd.to_numeric(df['premium'], errors='coerce')
    df['spot_price'] = pd.to_numeric(df['spot_price'], errors='coerce')
    df.dropna(subset=['strike', 'premium', 'spot_price'], inplace=True)

    # Step 2: Scale down the index levels and strikes
    scale_factor = 1000
    df['strike'] = df['strike'] / scale_factor
    df['spot_price'] = df['spot_price'] / scale_factor

    # Step 3: Adjust premiums using the contract multiplier
    contract_multiplier = 100  # Replace with the actual multiplier
    df['premium'] = df['premium'] * contract_multiplier
    # Initialize an empty list to store results
    vix_results = []

    # Get the list of unique dates in the dataset
    unique_dates = df['time'].unique()

    # Iterate over each date
    for current_date in unique_dates:
        # Fetch the risk-free rate (r) for the current date
        r = fetch_interest(current_date)  # Replace with actual interest rate fetching logic

        # Filter data for the current date
        daily_options = df[df['time'] == current_date].copy()

        # Get the spot price for the current date
        spot_price = daily_options['spot_price'].iloc[0]

        # Step 2: Prepare the data for the current date
        # Calculate Time to Expiration in years
        daily_options['T'] = (daily_options['due_date'] - daily_options['time']).dt.days / 365

        # Remove options that have already expired or have zero time to expiration
        daily_options = daily_options[daily_options['T'] > 0]

        # Step 3: Select maturities bracketing 30 days
        daily_options['days_to_expiration'] = (daily_options['due_date'] - daily_options['time']).dt.days
        # Remove maturities that are expired (negative days)
        daily_options = daily_options[daily_options['days_to_expiration'] > 0]
        maturities = daily_options['due_date'].unique()
        maturities_days = [(maturity, (maturity - current_date).days) for maturity in maturities]
        # Sort maturities based on how close they are to 30 days
        maturities_sorted = sorted(maturities_days, key=lambda x: abs(x[1] - 30))
        # Select two maturities that bracket 30 days
        if len(maturities_sorted) < 2:
            print(f"Date {current_date.date()}: Not enough maturities to bracket 30 days.")
            continue  # Skip to the next date
        else:
            T1_date, T1_days = maturities_sorted[0]
            T2_date, T2_days = maturities_sorted[1]
            # Ensure T1 < T2
            if T1_days > T2_days:
                T1_date, T1_days, T2_date, T2_days = T2_date, T2_days, T1_date, T1_days

        # Get options for T1 and T2
        options_T1 = daily_options[daily_options['due_date'] == T1_date]
        options_T2 = daily_options[daily_options['due_date'] == T2_date]

        # Calculate variance for T1
        T1 = T1_days / 365
        sigma_squared_T1 = calculate_variance(options_T1, T1, r, spot_price)

        # Calculate variance for T2
        T2 = T2_days / 365
        sigma_squared_T2 = calculate_variance(options_T2, T2, r, spot_price)

        # Check if variances are calculated successfully
        if sigma_squared_T1 is None or sigma_squared_T2 is None:
            print(f"Date {current_date.date()}: Not enough data to calculate variance.")
            continue  # Skip to the next date

        # Interpolate to get 30-day variance
        T_30 = 30 / 365  # 30 days expressed in years

        # Correct interpolation formula
        sigma_squared_30 = (
            T1 * sigma_squared_T1 * ((T2 - T_30) / (T2 - T1)) +
            T2 * sigma_squared_T2 * ((T_30 - T1) / (T2 - T1))
        )

        # Ensure sigma_squared_30 is positive
        if sigma_squared_30 <= 0:
            print(f"Date {current_date.date()}: Negative variance calculated.")
            continue  # Skip to the next date

        # Compute VIX
        VIX = 100 * np.sqrt(sigma_squared_30)

        # Append result to the list
        vix_results.append({'Date': current_date.date(), 'VIX': VIX})

        # Optional: print the result
        print(f"Date {current_date.date()}: The calculated VIX-equivalent index is {VIX:.2f}")

    # Convert results to a DataFrame
    vix_df = pd.DataFrame(vix_results)

    return vix_df

import pandas as pd
import numpy as np

## NOT USED
def calculate_daily_vix(df):
    # Initialize an empty list to store results
    vix_results = []

    # Get the list of unique dates in the dataset
    unique_dates = df['time'].unique()

    for current_date in unique_dates:
        # Filter data for the current date
        daily_options = df[df['time'] == current_date].copy()
        spot_price = daily_options['spot_price'].iloc[0]

        # Ensure 'due_date' and 'time' are datetime and tz-naive
        daily_options['due_date'] = pd.to_datetime(daily_options['due_date']).dt.tz_localize(None)
        daily_options['time'] = pd.to_datetime(daily_options['time']).dt.tz_localize(None)

        # Calculate Days to Maturity if not already calculated
        daily_options['days_to_maturity'] = (daily_options['due_date'] - daily_options['time']).dt.days

        # Group options by maturity
        maturities = daily_options['due_date'].unique()

        variances = []
        times = []

        for maturity in maturities:
            options = daily_options[daily_options['due_date'] == maturity].copy()
            T = options['days_to_maturity'].iloc[0] / 365
            times.append(T)
            from BrAPIWrapper import BrAPIWrapper
            api = BrAPIWrapper()
            # Fetch or calculate risk-free rate for the date and maturity
            # Get prime rate data for the current date
            prime_rate_data = api.get_prime_rate(start='01/01/2022', end='01/06/2024')

            # Match the date and get the rate
            r = prime_rate_data[prime_rate_data['date'].dt.date == current_date.date()]['rate'].iloc[0]

            # Calculate forward index level F
            F = spot_price * np.exp(r * T)

            # Determine K0 (strike price immediately below the forward price)
            options['strike'] = options['strike'].astype(float)
            K0 = options[options['strike'] <= F]['strike'].max()

            # Handle case where no strike is below F
            if pd.isna(K0):
                K0 = options['strike'].min()

            # Calculate ΔK
            options = options.sort_values('strike')
            strikes = options['strike'].unique()

            if len(strikes) < 2:
                # Not enough strikes to compute delta_K properly
                print(f"Skipping maturity {maturity.date()} on date {current_date.date()} due to insufficient strikes.")
                continue
            else:
                delta_K = {}
                for i, K in enumerate(strikes):
                    if i == 0:
                        delta_K[K] = strikes[i+1] - K
                    elif i == len(strikes) - 1:
                        delta_K[K] = K - strikes[i-1]
                    else:
                        delta_K[K] = (strikes[i+1] - strikes[i-1]) / 2

            options['deltaK'] = options['strike'].map(delta_K)

            # Calculate the variance contribution for each option
            options['variance_contribution'] = (
                (2 * options['deltaK'] / options['strike'] ** 2) *
                np.exp(r * T) *
                options['premium']
            )

            # Sum the variance contributions
            sigma_squared = options['variance_contribution'].sum()

            # Subtract the adjustment term
            sigma_squared -= (1 / T) * ((F / K0 - 1) ** 2)

            variances.append({'sigma_squared': sigma_squared, 'T': T})

        # Interpolate variance to 30 days
        if len(variances) >= 2:
            # Sort variances by T
            variances = sorted(variances, key=lambda x: x['T'])
            T1 = variances[0]['T']
            T2 = variances[1]['T']
            sigma_squared_T1 = variances[0]['sigma_squared']
            sigma_squared_T2 = variances[1]['sigma_squared']

            # Interpolate to 30-day variance
            N30 = 30 / 365
            sigma_squared_30 = (
                sigma_squared_T1 * (T2 - N30) / (T2 - T1) +
                sigma_squared_T2 * (N30 - T1) / (T2 - T1)
            )
        elif len(variances) == 1:
            # If only one variance is available, use it directly
            sigma_squared_30 = variances[0]['sigma_squared']
        else:
            # No variances were calculated for this date
            print(f"No variances calculated for date {current_date.date()}. Skipping VIX calculation.")
            continue

        # Ensure sigma_squared_30 is non-negative
        if sigma_squared_30 < 0:
            print(f"Negative variance calculated for date {current_date.date()}. Setting variance to zero.")
            sigma_squared_30 = 0

        # Calculate VIX
        VIX = 100 * np.sqrt(sigma_squared_30)

        # Append the result
        vix_results.append({'date': current_date, 'VIX': VIX})

    # Convert results to DataFrame
    vix_df = pd.DataFrame(vix_results)
    return vix_df
