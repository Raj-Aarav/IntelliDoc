# llm_clients/groq_client.py

import aiohttp

class GroqLLMClient:
    """Async client for Groq Cloud API."""
    
    def __init__(self, api_key: str, model: str, config):
        self.api_key = api_key
        self.model = model
        self.config = config
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(self, prompt: str) -> str:
        """Generate response using Groq API."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=60
                ) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        error_msg = data.get('error', {}).get('message', 'Unknown error')
                        return f"Error: {error_msg}"
                        
        except Exception as e:
            return f"Error generating response: {str(e)}"
