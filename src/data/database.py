"""数据库模块 - SQLAlchemy ORM"""
from sqlalchemy import create_engine, Column, String, Float, Date, Boolean, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from datetime import datetime

Base = declarative_base()


class Stock(Base):
    """股票基础信息表"""
    __tablename__ = 'stocks'
    
    symbol = Column(String(10), primary_key=True)
    name = Column(String(50))
    industry = Column(String(50))
    market = Column(String(10))
    is_st = Column(Boolean, default=False)
    is_chuangyeban = Column(Boolean, default=False)
    list_date = Column(Date)


class KlineDaily(Base):
    """日K线数据表"""
    __tablename__ = 'kline_daily'
    
    symbol = Column(String(10), primary_key=True)
    trade_date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    amount = Column(Float)
    amplitude = Column(Float)
    pct_change = Column(Float)
    change_amount = Column(Float)
    turnover = Column(Float)


class SentimentData(Base):
    """舆情数据表"""
    __tablename__ = 'sentiment_data'
    
    news_id = Column(String(64), primary_key=True)
    symbol = Column(String(10))
    title = Column(Text)
    content = Column(Text)
    source = Column(String(50))
    publish_time = Column(DateTime)
    category = Column(String(20))
    sentiment_score = Column(Float)
    keywords = Column(Text)
    url = Column(Text)
    summary = Column(Text)
    
    # 向量化关联层级
    policy_tags = Column(Text)
    industry_tags = Column(Text)
    company_tags = Column(Text)
    stock_tags = Column(Text)
    policy_relevance = Column(Float, default=0)
    industry_relevance = Column(Float, default=0)
    company_relevance = Column(Float, default=0)
    stock_relevance = Column(Float, default=0)
    
    collected_at = Column(DateTime, default=lambda: datetime.now)
    fetched_at = Column(DateTime, default=lambda: datetime.now)
    used_in_debate = Column(Boolean, default=False)


class PortfolioPosition(Base):
    """持仓记录表"""
    __tablename__ = 'portfolio_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    name = Column(String(50))
    shares = Column(Integer, nullable=False, default=lambda: 0)
    quantity = Column(Integer, nullable=False, default=lambda: 0)
    entry_price = Column(Float, nullable=False, default=lambda: 0.0)
    buy_price = Column(Float, nullable=False, default=lambda: 0.0)
    entry_date = Column(DateTime, nullable=False, default=lambda: datetime.now())
    buy_date = Column(Date, nullable=False)
    current_price = Column(Float)
    prev_close = Column(Float)
    day_high = Column(Float)
    day_low = Column(Float)
    highest_price = Column(Float)
    lowest_price = Column(Float)
    current_value = Column(Float)
    profit_loss = Column(Float)
    profit_loss_pct = Column(Float)
    realized_pnl = Column(Float, default=lambda: 0.0)
    realized_pnl_pct = Column(Float, default=lambda: 0.0)
    weight = Column(Float)
    sector = Column(String(50))
    status = Column(String(20), default=lambda: '持有中')
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now())
    updated_at = Column(DateTime, default=lambda: datetime.now(), onupdate=datetime.now)


class PortfolioHistory(Base):
    """持仓历史记录表"""
    __tablename__ = 'portfolio_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    total_value = Column(Float)
    total_cost = Column(Float)
    total_profit_loss = Column(Float)
    cash_balance = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now())


class BacktestResult(Base):
    """回测结果表"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50))
    start_date = Column(Date)
    end_date = Column(Date)
    initial_capital = Column(Float)
    final_value = Column(Float)
    total_return = Column(Float)
    annual_return = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    parameters = Column(Text)
    params = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now())


class TradeRecord(Base):
    """交易记录表"""
    __tablename__ = 'trade_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer)
    symbol = Column(String(10))
    entry_date = Column(DateTime)
    exit_date = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    shares = Column(Integer)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    holding_days = Column(Integer)
    exit_reason = Column(String(50))


class AnalysisTask(Base):
    """分析任务表"""
    __tablename__ = 'analysis_tasks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10))
    task_type = Column(String(20))
    status = Column(String(20))
    scheduled_at = Column(DateTime)
    parameters = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now())


class AnalysisResult(Base):
    """分析结果表"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer)
    symbol = Column(String(10))
    result_type = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now())


class Database:
    """数据库管理器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径，默认使用项目目录下的data/a_stock.db
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "a_stock.db"
        
        # 确保目录存在
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """获取数据库会话"""
        return self.Session()
    
    def _parse_date(self, val):
        """将各种日期格式转换为Python date对象"""
        if val is None or val == '' or val == '-':
            return None
        if isinstance(val, str):
            for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
                try:
                    from datetime import datetime
                    return datetime.strptime(val, fmt).date()
                except ValueError:
                    continue
        from datetime import date as date_type
        if isinstance(val, date_type):
            return val
        try:
            from datetime import datetime
            return pd.to_datetime(val).date()
        except Exception:
            return None

    def save_kline(self, symbol: str, df):
        """
        保存K线数据
        
        Args:
            symbol: 股票代码
            df: DataFrame，包含日K线数据
        """
        session = self.get_session()
        try:
            for _, row in df.iterrows():
                # 处理列名映射
                kline = KlineDaily(
                    symbol=symbol,
                    trade_date=self._parse_date(row.get('日期', row.get('trade_date'))),
                    open=row.get('开盘', row.get('open')),
                    high=row.get('最高', row.get('high')),
                    low=row.get('最低', row.get('low')),
                    close=row.get('收盘', row.get('close')),
                    volume=row.get('成交量', row.get('volume')),
                    amount=row.get('成交额', row.get('amount')),
                    amplitude=row.get('振幅', row.get('amplitude')),
                    pct_change=row.get('涨跌幅', row.get('pct_change')),
                    change_amount=row.get('涨跌额', row.get('change_amount')),
                    turnover=row.get('换手率', row.get('turnover'))
                )
                session.merge(kline)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_kline(self, symbol: str, start_date: str = None, end_date: str = None):
        """
        获取K线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (可选)
            end_date: 结束日期 (可选)
        
        Returns:
            DataFrame
        """
        import pandas as pd
        
        session = self.get_session()
        try:
            query = session.query(KlineDaily).filter(KlineDaily.symbol == symbol)
            
            if start_date:
                sd = self._parse_date(start_date)
                if sd:
                    query = query.filter(KlineDaily.trade_date >= sd)
            if end_date:
                ed = self._parse_date(end_date)
                if ed:
                    query = query.filter(KlineDaily.trade_date <= ed)
            
            results = query.order_by(KlineDaily.trade_date).all()
            
            if not results:
                return pd.DataFrame()
            
            data = []
            for r in results:
                data.append({
                    'symbol': r.symbol,
                    'trade_date': r.trade_date,
                    'open': r.open,
                    'high': r.high,
                    'low': r.low,
                    'close': r.close,
                    'volume': r.volume,
                    'amount': r.amount,
                    'amplitude': r.amplitude,
                    'pct_change': r.pct_change,
                    'change_amount': r.change_amount,
                    'turnover': r.turnover
                })
            
            return pd.DataFrame(data)
        finally:
            session.close()

    def save_stock(self, session, stock_data: dict):
        """
        保存股票基础信息

        Args:
            session: 数据库会话
            stock_data: 股票数据字典，包含 symbol, name, industry, market, is_st, is_chuangyeban, list_date
        """
        stock = session.query(Stock).filter(Stock.symbol == stock_data.get('symbol')).first()
        if stock is None:
            stock = Stock(symbol=stock_data.get('symbol'))
        stock.name = stock_data.get('name', stock.name)
        stock.industry = stock_data.get('industry', stock.industry)
        stock.market = stock_data.get('market', stock.market)
        stock.is_st = stock_data.get('is_st', stock.is_st)
        stock.is_chuangyeban = stock_data.get('is_chuangyeban', stock.is_chuangyeban)
        stock.list_date = stock_data.get('list_date', stock.list_date)
        session.merge(stock)
        session.commit()

    def get_stock(self, session, symbol: str):
        """
        获取股票信息

        Args:
            session: 数据库会话
            symbol: 股票代码

        Returns:
            Stock对象或None
        """
        return session.query(Stock).filter(Stock.symbol == symbol).first()

    def save_backtest_result(self, session, report_data: dict):
        """
        保存回测结果

        Args:
            session: 数据库会话
            report_data: 回测结果字典，包含 strategy_name, start_date, end_date,
                         initial_capital, final_value, total_return, annual_return,
                         max_drawdown, sharpe_ratio, win_rate, total_trades, parameters
        """
        import json
        result = BacktestResult(
            strategy_name=report_data.get('strategy_name'),
            start_date=report_data.get('start_date'),
            end_date=report_data.get('end_date'),
            initial_capital=report_data.get('initial_capital'),
            final_value=report_data.get('final_value'),
            total_return=report_data.get('total_return'),
            annual_return=report_data.get('annual_return'),
            max_drawdown=report_data.get('max_drawdown'),
            sharpe_ratio=report_data.get('sharpe_ratio'),
            win_rate=report_data.get('win_rate'),
            total_trades=report_data.get('total_trades'),
            parameters=json.dumps(report_data.get('parameters', {}), ensure_ascii=False) if isinstance(report_data.get('parameters'), dict) else report_data.get('parameters'),
            params=json.dumps(report_data.get('params', {}), ensure_ascii=False) if isinstance(report_data.get('params'), dict) else report_data.get('params'),
        )
        session.add(result)
        session.commit()
        return result

    def save_trade_record(self, session, trade_data: dict):
        """
        保存交易记录

        Args:
            session: 数据库会话
            trade_data: 交易记录字典，包含 backtest_id, symbol, entry_date, exit_date,
                         entry_price, exit_price, shares, pnl, pnl_pct, holding_days, exit_reason
        """
        record = TradeRecord(
            backtest_id=trade_data.get('backtest_id'),
            symbol=trade_data.get('symbol'),
            entry_date=trade_data.get('entry_date'),
            exit_date=trade_data.get('exit_date'),
            entry_price=trade_data.get('entry_price'),
            exit_price=trade_data.get('exit_price'),
            shares=trade_data.get('shares'),
            pnl=trade_data.get('pnl'),
            pnl_pct=trade_data.get('pnl_pct'),
            holding_days=trade_data.get('holding_days'),
            exit_reason=trade_data.get('exit_reason'),
        )
        session.add(record)
        session.commit()
        return record

    def get_backtest_results(self, session, limit=10):
        """
        获取最近的回测结果

        Args:
            session: 数据库会话
            limit: 返回结果数量，默认10

        Returns:
            BacktestResult列表
        """
        return session.query(BacktestResult).order_by(BacktestResult.created_at.desc()).limit(limit).all()



# 全局数据库实例
_db_instance = None

def get_db():
    """获取数据库会话生成器"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    session = _db_instance.get_session()
    try:
        yield session
    finally:
        session.close()

def init_db():
    """初始化数据库 (创建所有表)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    # create_all 在 Database.__init__ 中已经调用
    print("数据库初始化完成")

# 兼容旧代码
db = Database()
