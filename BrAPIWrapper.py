import os
import requests
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import time
from datetime import datetime, timezone, timedelta
from utils import historical_data_to_dataframe, financial_statement_to_dataframe, prime_rate_to_dataframe, inflation_to_dataframe
from typing import Dict, Any

class BrAPIError(Exception):
    """Custom exception for BRAPI API errors"""
    def __init__(self, message="BRAPI API error"):
        self.message = message
        super().__init__(self.message)
    pass

class BrAPIWrapper:
    """
    A wrapper class for the BRAPI API, providing methods to fetch various financial data.
    """

    BASE_URL = "https://brapi.dev/api"

    def __init__(self, token: Optional[str] = None, rate_limit: float = 1.0):
        self.token = token or os.getenv('BRAPI_API_KEY')
        if not self.token:
            raise ValueError("API token not provided and BRAPI_API_KEY not found in environment variables")
        self.rate_limit = rate_limit
        self.last_request_time = 0

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the BRAPI API with rate limiting.

        :param endpoint: API endpoint
        :param params: Query parameters
        :return: JSON response from the API
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params['token'] = self.token

        # Implement rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise BrAPIError(f"API request failed: {str(e)}")

    @staticmethod
    def _format_date(date: str) -> str:
        """
        Format date string to DD/MM/YYYY.

        :param date: Date string in any format
        :return: Formatted date string
        """
        try:
            return datetime.strptime(date, "%d/%m/%Y").strftime("%d/%m/%Y")
        except ValueError:
            raise ValueError("Invalid date format. Use DD/MM/YYYY.")

    def get_quote(self, 
                  tickers: Union[str, List[str]], 
                  range: str = '5d', 
                  interval: str = '1d', 
                  fundamental: bool = True, 
                  dividends: bool = False, 
                  modules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch quote data for given ticker(s).

        :param tickers: Single ticker symbol or list of ticker symbols
        :param range: Time range for historical data
        :param interval: Interval for historical data
        :param fundamental: Include fundamental data
        :param dividends: Include dividend data
        :param modules: Additional modules to include
        :return: Dictionary containing the API response
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        params = {
            'range': range,
            'interval': interval,
            'fundamental': str(fundamental).lower(),
            'dividends': str(dividends).lower(),
        }
        
        if modules:
            params['modules'] = ','.join(modules)

        return self._make_request(f"quote/{','.join(tickers)}", params)
    

    def get_historical_data(self, ticker: str, range: str = '5d') -> List[Dict[str, Any]]:
        """
        Extract historical price data for a given ticker.

        :param ticker: Ticker symbol
        :param range: Time range for historical data
        :return: List of dictionaries containing historical price data
        """
        data = self.get_quote(ticker, range=range)
        return data['results'][0]['historicalDataPrice']

    # still to implement code to parse results
    def get_dividends(self, ticker: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract dividend data for a given ticker.

        :param ticker: Ticker symbol
        :return: Dictionary containing cash dividends and stock dividends
        """
        data = self.get_quote([ticker], dividends=True)
        return data['results'][0]['dividendsData']

    def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> List[Dict[str, Any]]:
        """
        Extract balance sheet data for a given ticker.

        :param ticker: Ticker symbol
        :param quarterly: If True, fetch quarterly data instead of annual
        :return: List of dictionaries containing balance sheet data
        """
        module = 'balanceSheetHistoryQuarterly' if quarterly else 'balanceSheetHistory'
        data = self.get_quote([ticker], modules=[module])
        raw = data['results'][0][module]['balanceSheetStatements']
        return  financial_statement_to_dataframe(raw)

    def get_income_statement(self, ticker: str, quarterly: bool = False) -> List[Dict[str, Any]]:
        """
        Extract income statement data for a given ticker.

        :param ticker: Ticker symbol
        :param quarterly: If True, fetch quarterly data instead of annual
        :return: List of dictionaries containing income statement data
        """
        module = 'incomeStatementHistoryQuarterly' if quarterly else 'incomeStatementHistory'
        data = self.get_quote([ticker], modules=[module])
        raw = data['results'][0][module]['incomeStatementHistory']
        return  financial_statement_to_dataframe(raw)

    # returns json
    def get_summary_profile(self, ticker: str) -> Dict[str, Any]:
        """
        Extract summary profile data for a given ticker.

        :param ticker: Ticker symbol
        :return: Dictionary containing summary profile data
        """
        data = self.get_quote([ticker], modules=['summaryProfile'])
        return data['results'][0]['summaryProfile']
    
    # returns json
    def get_key_statistics(self, ticker: str) -> Dict[str, Any]:
        """
        Extract key statistics for a given ticker.

        :param ticker: Ticker symbol
        :return: Dictionary containing key statistics
        """
        data = self.get_quote([ticker], modules=['defaultKeyStatistics'])
        return data['results'][0]['defaultKeyStatistics']

    
    def search_tickers(self, 
                       search: Optional[str] = None, 
                       sort_by: str = 'name', 
                       sort_order: str = 'asc', 
                       limit: int = 50, 
                       page: int = 1, 
                       ticker_type: Optional[str] = None, 
                       sector: Optional[str] = None) -> pd.DataFrame:
        """
        Search for available tickers and return results as a pandas DataFrame.

        :param search: Search query for specific tickers
        :param sort_by: Field to sort by (e.g., 'name', 'close', 'change', 'volume', 'market_cap_basic')
        :param sort_order: Sort order ('asc' or 'desc')
        :param limit: Number of results per page
        :param page: Page number of results
        :param ticker_type: Type of ticker ('stock', 'fund', or 'bdr')
        :param sector: Sector to filter by
        :return: pandas DataFrame with ticker information
        """
        params = {
            'sortBy': sort_by,
            'sortOrder': sort_order,
            'limit': limit,
            'page': page
        }
        
        if search:
            params['search'] = search
        if ticker_type:
            params['type'] = ticker_type
        if sector:
            params['sector'] = sector

        data = self._make_request("quote/list", params)

        # Combine stocks and indexes into a single list
        all_tickers = data.get('stocks', []) + data.get('indexes', [])

        # Convert to DataFrame
        df = pd.DataFrame(all_tickers)

        # Reorder columns if they exist
        desired_order = ['stock', 'name', 'sector', 'close', 'change', 'volume', 'market_cap', 'type']
        df = df.reindex(columns=[col for col in desired_order if col in df.columns] + 
                                [col for col in df.columns if col not in desired_order])

        return df

    def get_crypto(self, 
                   coins: Union[str, List[str]], 
                   currency: str = 'BRL') -> Dict[str, Any]:
        """
        Fetch cryptocurrency data for given coin(s).

        :param coins: Single coin symbol or list of coin symbols (e.g., 'BTC' or ['BTC', 'ETH'])
        :param currency: Currency for price conversion (default is 'BRL')
        :return: Dictionary containing the API response
        """
        if isinstance(coins, str):
            coins = [coins]
        
        params = {
            'coin': ','.join(coins),
            'currency': currency,
        }

        return self._make_request("v2/crypto", params)

    def search_crypto(self, search: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Search for available cryptocurrencies.

        :param search: Optional search string to filter cryptocurrencies
        :return: Dictionary containing a list of available cryptocurrency symbols
        """
        params = {}
        if search:
            params['search'] = search

        return self._make_request("v2/crypto/available", params)

    ## USED
    def get_prime_rate(self, 
                       country: str = 'brazil',
                       historical: bool = True,
                       start: Optional[str] = (datetime.now() - timedelta(days=180)).strftime('%d/%m/%Y'),
                       end: Optional[str] = (datetime.now() - timedelta(days=1)).strftime('%d/%m/%Y'),
                       sort_by: str = 'date',
                       sort_order: str = 'desc') -> Dict[str, Any]:
        """
        Fetch prime rate (SELIC) data for a specified country and time period.

        :param country: Country to fetch data for (default is 'brazil')
        :param historical: Whether to fetch historical data (default is False)
        :param start: Start date for historical data (format: DD/MM/YYYY)
        :param end: End date for historical data (format: DD/MM/YYYY)
        :param sort_by: Field to sort by ('date' or 'value', default is 'date')
        :param sort_order: Sort order ('asc' or 'desc', default is 'desc')
        :return: Dictionary containing prime rate data
        """
        params = {
            'country': country,
            'historical': str(historical).lower(),
            'sortBy': sort_by,
            'sortOrder': sort_order
        }

        if historical:
            if start:
                params['start'] = self._format_date(start)
            if end:
                params['end'] = self._format_date(end)

        raw = self._make_request("v2/prime-rate", params)
        raw = prime_rate_to_dataframe(raw)
        raw['date'] = pd.to_datetime(raw.index)
        raw.reset_index(drop=True, inplace=True)  # Drop the index to avoid conflict
        raw.rename(columns={'prime_rate': 'rate'}, inplace=True)
        raw['rate'] = raw['rate']/ 100
        return raw

    def get_inflation(self, 
                      country: str = 'brazil',
                      historical: bool = True,
                      start: Optional[str] = (datetime.now() - timedelta(days=180)).strftime('%d/%m/%Y'),
                      end: Optional[str] = (datetime.now() - timedelta(days=1)).strftime('%d/%m/%Y'),
                      sort_by: str = 'date',
                      sort_order: str = 'desc') -> Dict[str, Any]:
        """
        Fetch inflation data for a specified country and time period.

        :param country: Country to fetch data for (default is 'brazil')
        :param historical: Whether to fetch historical data (default is False)
        :param start: Start date for historical data (format: DD/MM/YYYY)
        :param end: End date for historical data (format: DD/MM/YYYY)
        :param sort_by: Field to sort by ('date' or 'value', default is 'date')
        :param sort_order: Sort order ('asc' or 'desc', default is 'desc')
        :return: Dictionary containing inflation data
        """
        params = {
            'country': country,
            'historical': str(historical).lower(),
            'sortBy': sort_by,
            'sortOrder': sort_order
        }

        if historical:
            if start:
                params['start'] = self._format_date(start)
            if end:
                params['end'] = self._format_date(end)

        raw = self._make_request("v2/inflation", params)
        inflation = inflation_to_dataframe(raw)
        return inflation

    def get_available_prime_rates(self, search: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch available prime rates, optionally filtered by a search term.

        :param search: Optional search term to filter available prime rates
        :return: Dictionary containing available prime rate data
        """
        endpoint = f"{self.BASE_URL}/v2/prime-rate/available"
        params = {'token': self.token}
        if search:
            params['search'] = search

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Handle 404 error (might mean no results found)
                return {"countries": []}
            else:
                raise BrAPIError(f"API request failed: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise BrAPIError(f"API request failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    api = BrAPIWrapper()  # Will use BRAPI_API_KEY from environment variables
    
    try:
        # Test get_quote with a single ticker
        quote_data = api.get_quote("PETR4")
        print("Quote data retrieved successfully")
        print(quote_data)
    except BrAPIError as e:
        print(f"Error in get_quote: {e}")

    try:
        # Test get_historical_data
        historical_data = api.get_historical_data("PETR4")
        print("Historical data retrieved successfully")
        print(historical_data)
    except BrAPIError as e:
        print(f"Error in get_historical_data: {e}")