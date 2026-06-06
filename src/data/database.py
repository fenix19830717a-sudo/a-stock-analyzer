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
    
    collected_at = Column(DateTime, default=datetime.now)
    fetched_at = Column(DateTime, default=datetime.now)
    used_in_debate = Column(Boolean, default=False)


class PortfolioPosition(Base):
    """持仓记录表"""
    __tablename__ = 'portfolio_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    name = Column(String(50))
    shares = Column(Integer, nullable=False, default=0)
    quantity = Column(Integer, nullable=False, default=0)
    entry_price = Column(Float, nullable=False, default=0.0)
    buy_price = Column(Float, nullable=False, default=0.0)
    entry_date = Column(DateTime, nullable=False, default=datetime.now)
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
    realized_pnl = Column(Float, default=0.0)
    realized_pnl_pct = Column(Float, default=0.0)
    weight = Column(Float)
    sector = Column(String(50))
    status = Column(String(20), default='持有中')
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PortfolioHistory(Base):
    """持仓历史记录表"""
    __tablename__ = 'portfolio_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    total_value = Column(Float)
    total_cost = Column(Float)
    total_profit_loss = Column(Float)
    cash_balance = Column(Float)
    created_at = Column(DateTime, default=datetime.now)


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
    created_at = Column(DateTime, default=datetime.now)


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
    created_at = Column(DateTime, default=datetime.now)


class AnalysisResult(Base):
    """分析结果表"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer)
    symbol = Column(String(10))
    result_type = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)


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
                    trade_date=row.get('日期', row.get('trade_date')),
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
                query = query.filter(KlineDaily.trade_date >= start_date)
            if end_date:
                query = query.filter(KlineDaily.trade_date <= end_date)
            
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
