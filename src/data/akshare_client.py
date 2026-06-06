"""AkShare数据源客户端"""
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 尝试导入akshare，如果失败则使用Mock
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("akshare未安装，使用Mock数据模式")


class AkShareClient:
    """AkShare数据源客户端 - 主数据源"""
    
    def __init__(self):
        self.name = "akshare"
        self._cache = {}
    
    def get_stock_list(self):
        """
        获取A股列表
        
        Returns:
            DataFrame: 包含代码、名称、行业
        """
        if AKSHARE_AVAILABLE:
            try:
                df = ak.stock_zh_a_spot_em()
                return df[['代码', '名称', '所属行业']].rename(columns={
                    '代码': 'symbol',
                    '名称': 'name',
                    '所属行业': 'industry'
                })
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
        
        # Mock数据
        return pd.DataFrame({
            'symbol': ['600519', '000858', '002594', '300750', '601318'],
            'name': ['贵州茅台', '五粮液', '比亚迪', '宁德时代', '中国平安'],
            'industry': ['白酒', '白酒', '汽车', '电池', '保险']
        })
    
    def get_kline(self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq"):
        """
        获取历史K线数据
        
        Args:
            symbol: 股票代码 (如 "000001")
            start_date: 开始日期 (如 "20200101")
            end_date: 结束日期 (如 "20260606")
            adjust: 复权方式 ("qfq"=前复权, "hfq"=后复权, ""=不复权)
        
        Returns:
            DataFrame: K线数据
        """
        if AKSHARE_AVAILABLE:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
                time.sleep(random.uniform(0.5, 1.5))
                return df
            except Exception as e:
                logger.error(f"获取{symbol}K线失败: {e}")
        
        # Mock数据生成
        return self._generate_mock_kline(symbol, start_date, end_date)
    
    def _generate_mock_kline(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟K线数据"""
        # 解析日期
        try:
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')
        except:
            start = datetime(2023, 1, 1)
            end = datetime(2024, 12, 31)
        
        # 生成交易日
        dates = pd.date_range(start=start, end=end, freq='B')  # 工作日
        
        # 基于symbol生成确定性随机价格
        np.random.seed(hash(symbol) % 2**32)
        base_price = 100 + (hash(symbol) % 100)
        
        # 生成价格序列 (随机游走)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            '日期': dates.strftime('%Y-%m-%d'),
            '开盘': prices * (1 + np.random.randn(len(dates)) * 0.005),
            '最高': prices * (1 + abs(np.random.randn(len(dates))) * 0.015),
            '最低': prices * (1 - abs(np.random.randn(len(dates))) * 0.015),
            '收盘': prices,
            '成交量': np.random.randint(1000000, 10000000, len(dates)),
            '成交额': np.random.randint(100000000, 1000000000, len(dates)),
            '振幅': abs(np.random.randn(len(dates))) * 3,
            '涨跌幅': returns * 100,
            '涨跌额': prices * returns,
            '换手率': np.random.uniform(0.5, 5, len(dates))
        })
        
        return df
    
    def get_realtime_spot(self):
        """
        获取实时行情
        
        Returns:
            DataFrame: 全市场实时行情
        """
        if AKSHARE_AVAILABLE:
            try:
                return ak.stock_zh_a_spot_em()
            except Exception as e:
                logger.error(f"获取实时行情失败: {e}")
        
        # Mock数据
        return pd.DataFrame({
            '代码': ['600519', '000858', '002594', '300750', '601318'],
            '名称': ['贵州茅台', '五粮液', '比亚迪', '宁德时代', '中国平安'],
            '最新价': [1680.5, 145.3, 245.6, 198.5, 42.35],
            '涨跌幅': [1.2, 0.8, -0.5, 2.1, -0.3]
        })
    
    def get_industry_list(self):
        """
        获取行业分类列表
        
        Returns:
            DataFrame: 行业分类
        """
        if AKSHARE_AVAILABLE:
            try:
                return ak.stock_board_industry_name_ths()
            except Exception as e:
                logger.error(f"获取行业分类失败: {e}")
        
        return pd.DataFrame({
            '行业': ['白酒', '汽车', '电池', '保险', '银行']
        })
    
    def get_index_kline(self, symbol: str = "000001", start_date: str = None, end_date: str = None):
        """
        获取指数K线数据
        
        Args:
            symbol: 指数代码 (如 "000001"=上证指数)
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame: 指数K线数据
        """
        if AKSHARE_AVAILABLE:
            try:
                df = ak.index_zh_a_hist(symbol=symbol, period="daily")
                if start_date:
                    df = df[df['日期'] >= start_date]
                if end_date:
                    df = df[df['日期'] <= end_date]
                return df
            except Exception as e:
                logger.error(f"获取指数{symbol}K线失败: {e}")
        
        return self._generate_mock_kline(symbol, start_date or '20230101', end_date or '20241231')


# 导入numpy用于mock数据生成
import numpy as np
