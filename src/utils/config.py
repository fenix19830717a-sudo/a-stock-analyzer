"""配置管理模块"""
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """配置管理器 - 单例模式"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        load_dotenv()
        
        # 查找配置文件
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if not config_path.exists():
            config_path = Path("config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 从环境变量覆盖敏感信息
        self._config['llm']['api_key'] = os.getenv('OPENAI_API_KEY', '')
    
    def get(self, key_path: str, default=None):
        """
        通过点号路径获取配置
        
        Args:
            key_path: 如 'app.name', 'hmm.market_n_components'
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_all(self):
        """获取全部配置"""
        return self._config


# 全局配置实例
config = Config()
