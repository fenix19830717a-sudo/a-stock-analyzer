"""股票筛选器"""
import pandas as pd
from typing import Dict


class StockScreener:
    """
    股票筛选器
    
    默认排除:
    - 创业板 (300/301开头)
    - ST股
    - 价格低于N元的个股
    """
    
    DEFAULT_CONFIG = {
        'exclude_chuangyeban': True,
        'exclude_st': True,
        'min_price': 5.0,
        'min_market_cap': 10e8,
        'min_list_days': 60
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or self.DEFAULT_CONFIG.copy()
    
    def filter(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        应用筛选条件
        
        Args:
            stocks_df: DataFrame，包含股票列表
        
        Returns:
            DataFrame: 筛选后的股票列表
        """
        filtered = stocks_df.copy()
        
        # 排除创业板
        if self.config.get('exclude_chuangyeban', True):
            filtered = filtered[~filtered['symbol'].astype(str).str.startswith('300')]
            filtered = filtered[~filtered['symbol'].astype(str).str.startswith('301')]
        
        # 排除ST股
        if self.config.get('exclude_st', True):
            if 'name' in filtered.columns:
                filtered = filtered[~filtered['name'].str.contains('ST', na=False)]
                filtered = filtered[~filtered['name'].str.contains('*ST', na=False)]
        
        # 最低股价
        if 'close' in filtered.columns and self.config.get('min_price'):
            filtered = filtered[filtered['close'] >= self.config['min_price']]
        
        # 最小市值
        if 'market_cap' in filtered.columns and self.config.get('min_market_cap'):
            filtered = filtered[filtered['market_cap'] >= self.config['min_market_cap']]
        
        return filtered
    
    def filter_by_technical(self, stocks_df: pd.DataFrame, 
                           technical_conditions: Dict) -> pd.DataFrame:
        """
        根据技术指标筛选
        
        Args:
            stocks_df: DataFrame
            technical_conditions: 技术指标条件
        
        Returns:
            DataFrame: 筛选结果
        """
        filtered = stocks_df.copy()
        
        # 换手率范围
        if 'turnover_range' in technical_conditions:
            min_turnover, max_turnover = technical_conditions['turnover_range']
            if 'turnover' in filtered.columns:
                filtered = filtered[
                    (filtered['turnover'] >= min_turnover) &
                    (filtered['turnover'] <= max_turnover)
                ]
        
        # 市盈率范围
        if 'pe_range' in technical_conditions:
            min_pe, max_pe = technical_conditions['pe_range']
            if 'pe' in filtered.columns:
                filtered = filtered[
                    (filtered['pe'] >= min_pe) &
                    (filtered['pe'] <= max_pe)
                ]
        
        return filtered
