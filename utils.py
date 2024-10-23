# Utils for other files

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any
from datetime import datetime, timedelta


def historical_data_to_dataframe(historical_data):
    """
    Convert historical data from JSON to a pandas DataFrame.
    
    :param historical_data: List of dictionaries containing historical data
    :return: pandas DataFrame with date as index and other fields as columns
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Convert Unix timestamp to datetime
    df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
    
    # Set 'date' as the index
    df.set_index('date', inplace=True)
    
    # Sort index in descending order (most recent date first)
    df.sort_index(ascending=False, inplace=True)
    
    # Convert UTC time to local time
    df.index = df.index.tz_convert('America/Sao_Paulo')
    
    return df

def financial_statement_to_dataframe(financial_data):
    """
    Convert financial statement data from JSON to a pandas DataFrame.
    
    :param financial_data: List of dictionaries containing financial statement data
    :return: pandas DataFrame with end dates as column headers and financial statement lines as index
    """
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(financial_data)
    
    # Convert endDate to datetime and format it
    df['endDate'] = pd.to_datetime(df['endDate'])
    df['endDate'] = df['endDate'].dt.strftime('%Y-%m-%d')
    
    # Set endDate as index temporarily
    df.set_index('endDate', inplace=True)
    
    # Transpose the DataFrame
    df_transposed = df.transpose()
    
    # Sort columns in descending order (most recent date first)
    df_transposed = df_transposed.sort_index(axis=1, ascending=False)
    
    # Format numbers (optional)
    df_transposed = df_transposed.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    
    return df_transposed

def prime_rate_to_dataframe(prime_rate_data):
    """
    Convert prime rate data to a pandas DataFrame with date as index and rate as value.
    
    :param prime_rate_data: Dictionary containing prime rate data
    :return: pandas DataFrame with date as index and prime rate as value
    """
    # Extract the list of prime rate entries
    prime_rates = prime_rate_data['prime-rate']
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(prime_rates)
    
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # Set 'date' as index
    df.set_index('date', inplace=True)
    
    # Sort index in descending order (most recent date first)
    df.sort_index(ascending=False, inplace=True)
    
    # Keep only the 'value' column and rename it
    df = df[['value']].rename(columns={'value': 'prime_rate'})
    
    # Convert prime_rate to float
    df['prime_rate'] = df['prime_rate'].astype(float)
    
    return df

def inflation_to_dataframe(inflation_rate_data):
    """
    Convert prime rate data to a pandas DataFrame with date as index and rate as value.
    
    :param prime_rate_data: Dictionary containing prime rate data
    :return: pandas DataFrame with date as index and prime rate as value
    """
    # Extract the list of prime rate entries
    inflation = inflation_rate_data['inflation']
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(inflation)
    
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # Set 'date' as index
    df.set_index('date', inplace=True)
    
    # Sort index in descending order (most recent date first)
    df.sort_index(ascending=False, inplace=True)
    
    # Keep only the 'value' column and rename it
    df = df[['value']].rename(columns={'value': 'prime_rate'})
    
    # Convert prime_rate to float
    df['prime_rate'] = df['prime_rate'].astype(float)
    
    return df

def get_price(quote_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract historical price data from quote data and create a pandas DataFrame.

        :param quote_data: Dictionary containing quote data returned by get_quote
        :return: pandas DataFrame with historical price data
        """
        # Extract historical price data
        historical_data = quote_data['results'][0]['historicalDataPrice']

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)

        # Convert Unix timestamp to UTC-3 datetime
        df['date'] = pd.to_datetime(df['date'], unit='s') - timedelta(hours=3)

        # Set date as index
        df.set_index('date', inplace=True)

        # Sort index in descending order (most recent date first)
        df.sort_index(ascending=False, inplace=True)

        # Select and reorder columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df