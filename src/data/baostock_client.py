"""BaoStock数据源客户端"""
import baostock as bs
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaoStockClient:
    """BaoStock数据源客户端 - 备用数据源"""
    
    def __init__(self):
        self.name = "baostock"
        self._logged_in = False
        self._login()
    
    def _login(self):
        """登录BaoStock"""
        try:
            result = bs.login()
            if result.error_code != '0':
                logger.error(f"BaoStock登录失败: {result.error_msg}")
                raise Exception(f"BaoStock登录失败: {result.error_msg}")
            self._logged_in = True
            logger.info("BaoStock登录成功")
        except Exception as e:
            logger.error(f"BaoStock登录异常: {e}")
            raise
    
    def get_kline(self, symbol: str, start_date: str, end_date: str, 
                  fields: str = "date,open,high,low,close,volume,amount"):
        """
        获取历史K线数据
        
        Args:
            symbol: 股票代码 (如 "000001")
            start_date: 开始日期 (如 "2020-01-01")
            end_date: 结束日期 (如 "2026-06-06")
            fields: 返回字段
        
        Returns:
            DataFrame: K线数据
        """
        if not self._logged_in:
            self._login()
        
        try:
            # 转换代码格式: 000001 -> sz.000001
            market = "sh" if symbol.startswith('6') else "sz"
            code = f"{market}.{symbol}"
            
            rs = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"  # 复权
            )
            
            if rs.error_code != '0':
                logger.error(f"获取{symbol}K线失败: {rs.error_msg}")
                raise Exception(f"获取{symbol}K线失败: {rs.error_msg}")
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 转换数据类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            logger.error(f"获取{symbol}K线异常: {e}")
            raise
    
    def get_stock_basic(self, symbol: str):
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码
        
        Returns:
            dict: 股票基本信息
        """
        if not self._logged_in:
            self._login()
        
        try:
            market = "sh" if symbol.startswith('6') else "sz"
            code = f"{market}.{symbol}"
            
            rs = bs.query_stock_basic(code=code)
            if rs.error_code != '0':
                logger.error(f"获取{symbol}基本信息失败: {rs.error_msg}")
                return {}
            
            if rs.next():
                return {
                    'code': rs.get_row_data()[0],
                    'code_name': rs.get_row_data()[1],
                    'ipo_date': rs.get_row_data()[2],
                    'out_date': rs.get_row_data()[3],
                    'type': rs.get_row_data()[4],
                    'status': rs.get_row_data()[5]
                }
            return {}
        except Exception as e:
            logger.error(f"获取{symbol}基本信息异常: {e}")
            return {}
    
    def __del__(self):
        """析构时登出"""
        if self._logged_in:
            try:
                bs.logout()
                logger.info("BaoStock登出")
            except:
                pass
