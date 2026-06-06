"""LLM客户端封装"""
import openai
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """LLM客户端封装"""
    
    def __init__(self):
        api_key = config.get('llm.api_key')
        if not api_key:
            logger.warning("未配置OpenAI API Key，LLM功能将不可用")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key)
        
        self.model = config.get('llm.model', 'gpt-4')
        self.temperature = config.get('llm.temperature', 0.7)
    
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用LLM生成回复
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
        
        Returns:
            str: LLM回复内容
        """
        if not self.client:
            logger.error("LLM客户端未初始化")
            return "[ERROR] LLM客户端未初始化，请配置OPENAI_API_KEY"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return f"[ERROR] {str(e)}"
