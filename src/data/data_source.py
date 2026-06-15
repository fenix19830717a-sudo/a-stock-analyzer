"""数据源管理器 - 主备自动降级"""
from src.data.akshare_client import AkShareClient
from src.data.database import Database, db
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataSourceManager:
    """
    数据源管理器 - 支持主备自动降级

    使用策略:
    1. 优先使用AkShare (主数据源)
    2. AkShare失败时降级到BaoStock (备数据源)
    3. 同时更新本地缓存
    """

    def __init__(self):
        self.primary = AkShareClient()
        self.backup = None
        try:
            from src.data.baostock_client import BaoStockClient
            self.backup = BaoStockClient()
            logger.info("BaoStock备数据源初始化完成")
        except Exception as e:
            logger.warning(f"BaoStock备数据源初始化失败，将仅使用AkShare: {e}")
        self.db = db

    def get_stock_list(self):
        """
        获取股票列表

        Returns:
            DataFrame: 股票列表
        """
        return self.primary.get_stock_list()


    def _normalize_columns(self, df):
        """标准化K线数据列名为英文"""
        if df is None or df.empty:
            return df
        column_map = {
            '日期': 'trade_date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change_amount',
            '换手率': 'turnover',
        }
        df = df.rename(columns=column_map)
        return df

    def get_kline(self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq"):
        """
        获取K线数据（带缓存和降级）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权方式

        Returns:
            DataFrame: K线数据
        """
        # 1. 检查数据库缓存
        cached_df = self.db.get_kline(symbol, start_date, end_date)
        if not cached_df.empty:
            logger.info(f"从缓存获取{symbol}K线数据")
            return self._normalize_columns(cached_df)

        # 2. 尝试主数据源 (AkShare)
        try:
            df = self.primary.get_kline(symbol, start_date, end_date, adjust)
            # 保存到数据库
            self.db.save_kline(symbol, df)
            logger.info(f"从AkShare获取{symbol}K线数据")
            return self._normalize_columns(df)
        except Exception as e:
            logger.warning(f"AkShare获取{symbol}失败: {e}, 尝试BaoStock")

        # 3. 降级到备数据源 (BaoStock)
        if self.backup is not None:
            try:
                # BaoStock日期格式不同，需要转换
                bs_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
                bs_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

                df = self.backup.get_kline(symbol, bs_start, bs_end)
                # 保存到数据库
                self.db.save_kline(symbol, df)
                logger.info(f"从BaoStock获取{symbol}K线数据")
                return df
            except Exception as e:
                logger.error(f"BaoStock也失败: {e}")
                raise DataSourceError(f"所有数据源不可用: {e}")
        else:
            raise DataSourceError("BaoStock备数据源不可用，且AkShare获取失败")

    def get_realtime_spot(self):
        """
        获取实时行情

        Returns:
            DataFrame: 实时行情
        """
        return self.primary.get_realtime_spot()

    def get_index_kline(self, symbol: str = "000001", start_date: str = None, end_date: str = None):
        """
        获取指数K线

        Args:
            symbol: 指数代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 指数K线数据
        """
        return self.primary.get_index_kline(symbol, start_date, end_date)


class DataSourceError(Exception):
    """数据源异常"""
    pass
