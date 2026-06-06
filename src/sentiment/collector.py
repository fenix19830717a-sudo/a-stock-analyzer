"""
舆情采集模块 - 新浪财经新闻抓取

功能:
1. 定期抓取新浪财经新闻
2. 提取政策-行业-企业-个股关联
3. 向量化存储
4. 供AI辩论参考
"""

import re
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, quote
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy.orm import Session

from src.data.database import get_db, SentimentData
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """新闻文章"""
    title: str
    content: str
    url: str
    source: str
    publish_time: datetime
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    # 向量化关联标签
    policy_tags: List[str] = field(default_factory=list)
    industry_tags: List[str] = field(default_factory=list)
    company_tags: List[str] = field(default_factory=list)
    stock_tags: List[str] = field(default_factory=list)
    # 关联度分数
    policy_relevance: float = 0.0
    industry_relevance: float = 0.0
    company_relevance: float = 0.0
    stock_relevance: float = 0.0


class SinaNewsCollector:
    """新浪财经新闻采集器"""

    BASE_URL = "https://finance.sina.com.cn"
    NEWS_API = "https://feed.sina.com.cn/api/roll/get"

    # 政策关键词库
    POLICY_KEYWORDS = [
        "政策", "监管", "央行", "证监会", "银保监会", "国务院", "发改委",
        "财政部", "货币政策", "财政政策", "降准", "降息", "LPR", "IPO",
        "注册制", "退市", "减持", "增持", "回购", "分红", "税收",
        "关税", "贸易", "中美", "一带一路", "双碳", "新能源", "芯片",
        "半导体", "人工智能", "AI", "数字经济", "基建", "房地产", "限购"
    ]

    # 行业关键词库
    INDUSTRY_KEYWORDS = {
        "银行": ["银行", "信贷", "存款", "贷款", "利率"],
        "证券": ["券商", "投行", "经纪", "自营"],
        "保险": ["保险", "保费", "寿险", "财险"],
        "房地产": ["房地产", "楼市", "房价", "拿地", "土拍"],
        "新能源": ["新能源", "光伏", "风电", "储能", "锂电池", "电动车"],
        "科技": ["科技", "软件", "互联网", "云计算", "大数据"],
        "医药": ["医药", "医疗", "创新药", "医保", "集采"],
        "消费": ["消费", "零售", "白酒", "食品饮料", "家电"],
        "制造": ["制造", "机械", "汽车", "军工", "航空"],
        "能源": ["能源", "石油", "天然气", "煤炭", "电力"]
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.logger = get_logger(self.__class__.__name__)

    def fetch_news(self, category: str = "stock", num: int = 50) -> List[NewsArticle]:
        """
        抓取新闻列表

        Args:
            category: 新闻类别 (stock/finance/economy)
            num: 获取数量
        """
        articles = []

        try:
            # 新浪财经滚动新闻API
            params = {
                'pageid': 153,
                'lid': 2516 if category == 'stock' else 2515,
                'num': num,
                'version': '1.2.4',
                'encode': 'utf-8'
            }

            response = self.session.get(self.NEWS_API, params=params, timeout=10)
            response.encoding = 'utf-8'
            data = response.json()

            if data.get('result', {}).get('data'):
                for item in data['result']['data']:
                    article = self._parse_news_item(item)
                    if article:
                        articles.append(article)

            self.logger.info(f"抓取新闻: {len(articles)}条")

        except Exception as e:
            self.logger.error(f"抓取新闻失败: {e}")

        return articles

    def _parse_news_item(self, item: Dict) -> Optional[NewsArticle]:
        """解析新闻条目"""
        try:
            title = item.get('title', '')
            url = item.get('url', '')
            publish_time = datetime.strptime(
                item.get('ctime', ''), '%Y-%m-%d %H:%M:%S'
            ) if 'ctime' in item else datetime.now()

            # 获取正文
            content = self._fetch_article_content(url)
            if not content:
                content = item.get('summary', '')

            article = NewsArticle(
                title=title,
                content=content,
                url=url,
                source="新浪财经",
                publish_time=publish_time,
                summary=item.get('summary', '')
            )

            # 提取关联标签
            self._extract_tags(article)

            return article

        except Exception as e:
            self.logger.warning(f"解析新闻失败: {e}")
            return None

    def _fetch_article_content(self, url: str) -> str:
        """获取文章正文"""
        try:
            response = self.session.get(url, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # 尝试多种正文选择器
            selectors = [
                '#artibody',
                '.article-content',
                '#article_content',
                '.content',
                'article'
            ]

            for selector in selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    # 清理脚本和样式
                    for script in content_div.find_all(['script', 'style']):
                        script.decompose()
                    return content_div.get_text(strip=True)

            return ""

        except Exception as e:
            self.logger.warning(f"获取正文失败 {url}: {e}")
            return ""

    def _extract_tags(self, article: NewsArticle):
        """提取向量化关联标签"""
        text = f"{article.title} {article.content}"

        # 政策标签提取
        policy_matches = []
        for keyword in self.POLICY_KEYWORDS:
            if keyword in text:
                policy_matches.append(keyword)
        article.policy_tags = policy_matches
        article.policy_relevance = min(1.0, len(policy_matches) / 3)

        # 行业标签提取
        industry_matches = []
        for industry, keywords in self.INDUSTRY_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count > 0:
                industry_matches.append(industry)
        article.industry_tags = industry_matches
        article.industry_relevance = min(1.0, len(industry_matches) / 2)

        # 企业/个股标签提取 (简化版，实际可用NER)
        article.company_tags = self._extract_companies(text)
        article.company_relevance = min(1.0, len(article.company_tags) / 2)

        # 股票代码提取
        article.stock_tags = self._extract_stock_codes(text)
        article.stock_relevance = min(1.0, len(article.stock_tags) / 3)

        article.keywords = list(set(
            article.policy_tags +
            article.industry_tags +
            article.company_tags +
            article.stock_tags
        ))

    def _extract_companies(self, text: str) -> List[str]:
        """提取企业名称 (简化规则)"""
        # 匹配 "XX公司"、"XX集团"、"XX股份" 等
        patterns = [
            r'([\u4e00-\u9fa5]{2,8})(?:公司|集团|股份|科技|企业)',
            r'([\u4e00-\u9fa5]{2,6})(?:银行|证券|保险|基金)'
        ]

        companies = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)

        return list(set(companies))[:10]

    def _extract_stock_codes(self, text: str) -> List[str]:
        """提取股票代码"""
        # 匹配 6位数字代码
        codes = re.findall(r'\b(\d{6})\b', text)
        # 过滤有效代码 (60/00/30开头)
        valid_codes = [c for c in codes if c[:2] in ['60', '00', '30', '68']]
        return list(set(valid_codes))[:5]


class EntityRelationExtractor:
    """
    实体关联提取器
    提取政策-行业-企业-个股四层关联
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def extract_relations(self, article: NewsArticle) -> Dict:
        """
        提取文章中的实体关联

        Returns:
            {
                'policy_industry': [(政策, 行业, 相关度)],
                'industry_company': [(行业, 企业, 相关度)],
                'company_stock': [(企业, 股票代码, 相关度)],
                'policy_stock': [(政策, 股票代码, 相关度)]
            }
        """
        relations = {
            'policy_industry': [],
            'industry_company': [],
            'company_stock': [],
            'policy_stock': []
        }

        # 政策-行业关联
        for policy in article.policy_tags:
            for industry in article.industry_tags:
                relevance = self._calculate_relevance(
                    policy, industry, article.content
                )
                if relevance > 0.3:
                    relations['policy_industry'].append((policy, industry, relevance))

        # 行业-企业关联
        for industry in article.industry_tags:
            for company in article.company_tags:
                relevance = self._calculate_relevance(
                    industry, company, article.content
                )
                if relevance > 0.3:
                    relations['industry_company'].append((industry, company, relevance))

        # 企业-股票关联
        for company in article.company_tags:
            for stock in article.stock_tags:
                relevance = self._calculate_relevance(
                    company, stock, article.content
                )
                if relevance > 0.3:
                    relations['company_stock'].append((company, stock, relevance))

        # 政策-股票直接关联
        for policy in article.policy_tags:
            for stock in article.stock_tags:
                relevance = self._calculate_relevance(
                    policy, stock, article.content
                )
                if relevance > 0.3:
                    relations['policy_stock'].append((policy, stock, relevance))

        return relations

    def _calculate_relevance(self, entity1: str, entity2: str, text: str) -> float:
        """计算两个实体在文本中的关联度"""
        # 简单实现：基于共现距离
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)

        if pos1 == -1 or pos2 == -1:
            return 0.0

        distance = abs(pos1 - pos2)
        text_len = len(text)

        # 距离越近，相关度越高
        relevance = max(0, 1 - distance / (text_len * 0.1))
        return min(1.0, relevance)


class VectorizedNewsStore:
    """
    向量化新闻存储
    支持按政策/行业/企业/个股查询
    """

    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session or next(get_db())
        self.logger = get_logger(self.__class__.__name__)

    def save_article(self, article: NewsArticle) -> bool:
        """保存文章到数据库"""
        try:
            # 生成唯一ID
            article_id = hashlib.md5(
                f"{article.title}{article.publish_time}".encode()
            ).hexdigest()

            # 检查是否已存在
            existing = self.db.query(SentimentData).filter(
                SentimentData.news_id == article_id
            ).first()
            if existing:
                return False

            sentiment = SentimentData(
                news_id=article_id,
                title=article.title,
                content=article.content[:2000],  # 限制长度
                url=article.url,
                source=article.source,
                publish_time=article.publish_time,
                summary=article.summary,
                keywords=','.join(article.keywords),
                policy_tags=','.join(article.policy_tags),
                industry_tags=','.join(article.industry_tags),
                company_tags=','.join(article.company_tags),
                stock_tags=','.join(article.stock_tags),
                policy_relevance=article.policy_relevance,
                industry_relevance=article.industry_relevance,
                company_relevance=article.company_relevance,
                stock_relevance=article.stock_relevance,
                collected_at=datetime.now()
            )

            self.db.add(sentiment)
            self.db.commit()
            return True

        except Exception as e:
            self.logger.error(f"保存文章失败: {e}")
            self.db.rollback()
            return False

    def query_by_stock(self, symbol: str, days: int = 7,
                       min_relevance: float = 0.3) -> List[SentimentData]:
        """
        按股票代码查询相关新闻

        Args:
            symbol: 股票代码
            days: 最近N天
            min_relevance: 最低相关度
        """
        start_date = datetime.now() - timedelta(days=days)

        results = self.db.query(SentimentData).filter(
            SentimentData.stock_tags.contains(symbol),
            SentimentData.publish_time >= start_date,
            SentimentData.stock_relevance >= min_relevance
        ).order_by(SentimentData.publish_time.desc()).all()

        return results

    def query_by_industry(self, industry: str, days: int = 7,
                          min_relevance: float = 0.3) -> List[SentimentData]:
        """按行业查询相关新闻"""
        start_date = datetime.now() - timedelta(days=days)

        results = self.db.query(SentimentData).filter(
            SentimentData.industry_tags.contains(industry),
            SentimentData.publish_time >= start_date,
            SentimentData.industry_relevance >= min_relevance
        ).order_by(SentimentData.publish_time.desc()).all()

        return results

    def query_by_policy(self, policy_keyword: str, days: int = 7,
                        min_relevance: float = 0.3) -> List[SentimentData]:
        """按政策关键词查询相关新闻"""
        start_date = datetime.now() - timedelta(days=days)

        results = self.db.query(SentimentData).filter(
            SentimentData.policy_tags.contains(policy_keyword),
            SentimentData.publish_time >= start_date,
            SentimentData.policy_relevance >= min_relevance
        ).order_by(SentimentData.publish_time.desc()).all()

        return results

    def query_related(self, symbol: str = None, industry: str = None,
                      policy: str = None, days: int = 7) -> List[SentimentData]:
        """
        综合查询 - 支持多维度关联

        查询逻辑:
        1. 如果提供股票代码，查询直接关联该股票的新闻
        2. 同时查询该股票所属行业的新闻
        3. 同时查询影响该股票的政策新闻
        4. 按综合相关度排序
        """
        results = []

        if symbol:
            stock_news = self.query_by_stock(symbol, days)
            results.extend(stock_news)

        if industry:
            industry_news = self.query_by_industry(industry, days)
            # 去重
            existing_ids = {n.news_id for n in results}
            for news in industry_news:
                if news.news_id not in existing_ids:
                    results.append(news)

        if policy:
            policy_news = self.query_by_policy(policy, days)
            existing_ids = {n.news_id for n in results}
            for news in policy_news:
                if news.news_id not in existing_ids:
                    results.append(news)

        # 按综合相关度排序
        results.sort(
            key=lambda x: (
                x.stock_relevance + x.industry_relevance +
                x.policy_relevance + x.company_relevance
            ),
            reverse=True
        )

        return results


class SentimentPipeline:
    """
    舆情处理流水线
    整合采集、提取、存储
    """

    def __init__(self, db_session: Optional[Session] = None):
        self.collector = SinaNewsCollector()
        self.extractor = EntityRelationExtractor()
        self.store = VectorizedNewsStore(db_session)
        self.logger = get_logger(self.__class__.__name__)

    def run(self, category: str = "stock", num: int = 50) -> Dict:
        """
        执行完整采集流程

        Returns:
            {'collected': int, 'saved': int, 'errors': int}
        """
        self.logger.info(f"开始舆情采集: category={category}, num={num}")

        # 采集
        articles = self.collector.fetch_news(category, num)

        saved = 0
        errors = 0

        for article in articles:
            try:
                # 提取关联
                relations = self.extractor.extract_relations(article)

                # 保存
                if self.store.save_article(article):
                    saved += 1

            except Exception as e:
                self.logger.warning(f"处理文章失败: {e}")
                errors += 1

        result = {
            'collected': len(articles),
            'saved': saved,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }

        self.logger.info(f"舆情采集完成: {result}")
        return result

    def get_context_for_debate(self, symbol: str, industry: str = None,
                                days: int = 3) -> str:
        """
        为AI辩论生成舆情上下文

        Returns:
            格式化的舆情摘要文本
        """
        news_list = self.store.query_related(
            symbol=symbol,
            industry=industry,
            days=days
        )

        if not news_list:
            return "近期无相关舆情信息。"

        # 按相关度选择Top 10
        top_news = news_list[:10]

        context_parts = [f"## 近期舆情信息 ({len(top_news)}条)"]

        for i, news in enumerate(top_news, 1):
            relevance_score = (
                news.stock_relevance * 0.4 +
                news.industry_relevance * 0.3 +
                news.policy_relevance * 0.3
            )

            context_parts.append(
                f"{i}. [{news.publish_time.strftime('%m-%d')}] "
                f"{news.title} "
                f"(相关度: {relevance_score:.2f})"
            )

            if news.summary:
                context_parts.append(f"   摘要: {news.summary[:100]}...")

        return '\n'.join(context_parts)
