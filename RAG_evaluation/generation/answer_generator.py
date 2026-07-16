import os
import json
import httpx

class LLMWrapper:
    """
    Base class for pluggable LLM wrappers.
    Any concrete subclass must implement the generate method.
    """
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement the generate method.")

class OllamaLLM(LLMWrapper):
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = httpx.post(url, json=payload, timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                return f"Error: Ollama returned status code {response.status_code}. Response: {response.text}"
        except Exception as e:
            return f"Error: Failed to connect to Ollama: {str(e)}"

class OpenAILLM(LLMWrapper):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: OpenAI API key is missing. Set OPENAI_API_KEY env var."
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return f"Error: OpenAI returned status code {response.status_code}. Response: {response.text}"
        except Exception as e:
            return f"Error: Failed to connect to OpenAI: {str(e)}"

class AnthropicLLM(LLMWrapper):
    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: Anthropic API key is missing. Set ANTHROPIC_API_KEY env var."
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                return data["content"][0]["text"].strip()
            else:
                return f"Error: Anthropic returned status code {response.status_code}. Response: {response.text}"
        except Exception as e:
            return f"Error: Failed to connect to Anthropic: {str(e)}"

class HuggingFaceLLM(LLMWrapper):
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        self.model_name = model_name
        self.pipeline = None
        
    def _lazy_init(self):
        if self.pipeline is None:
            from transformers import pipeline
            self.pipeline = pipeline("text-generation", model=self.model_name, device_map="auto")

    def generate(self, prompt: str) -> str:
        try:
            self._lazy_init()
            # Simple text generation logic
            results = self.pipeline(prompt, max_new_tokens=150, temperature=0.1, do_sample=True)
            generated_text = results[0]["generated_text"]
            # Extract generated response after prompt if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            return generated_text.strip()
        except Exception as e:
            return f"Error: HuggingFace pipeline failed: {str(e)}"

class FallbackLLM(LLMWrapper):
    """
    Fallback mock generator that extracts relevant text from context chunks 
    to construct a plausible answer. Ideal for offline, local testing without APIs/Ollama.
    """
    def generate(self, prompt: str) -> str:
        # Check if the prompt is for context precision evaluation
        if "relevance" in prompt and "true, false" in prompt:
            count = prompt.count("Document ")
            if count == 0:
                count = 5
            relevancies = [True] * min(2, count) + [False] * max(0, count - 2)
            relevance_str = ", ".join(["true" if r else "false" for r in relevancies])
            return f'{{\n  "relevance": [{relevance_str}]\n}}'
            
        # Check if the prompt is for faithfulness evaluation
        if "grounded/faithful" in prompt:
            return '{\n  "reasoning": "The generated statements are fully supported by the retrieved document text.",\n  "score": 0.95\n}'
            
        # Check if the prompt is for answer relevancy evaluation
        if "relevance of the generated answer" in prompt:
            return '{\n  "reasoning": "The response directly answers the question without redundant or extraneous info.",\n  "score": 0.90\n}'
            
        # Check if the prompt is for context recall evaluation
        if "Ground Truth answer and the retrieved documents" in prompt:
            return '{\n  "reasoning": "The key elements from the ground truth answer are present in the context chunks.",\n  "score": 0.92\n}'
            
        # Check if the prompt is for noise sensitivity evaluation
        if "Noise Sensitivity" in prompt or "misled, corrupted, or confused" in prompt:
            return '{\n  "reasoning": "The answer ignores irrelevant documents and draws only from the correct film information.",\n  "score": 0.05\n}'

        # Find context block and question for standard generation
        context_section = ""
        question_section = ""
        
        if "Context:" in prompt:
            parts = prompt.split("Context:")
            if len(parts) > 1:
                subparts = parts[1].split("Question:")
                context_section = subparts[0].strip()
                if len(subparts) > 1:
                    question_section = subparts[1].split("Answer:")[0].strip()
                    
        if not context_section:
            return "No context provided. Fallback mock answer."

            
        # Parse context lines
        lines = context_section.split("\n")
        
        # Simple extraction heuristics based on query keyword matching
        keywords = [w.lower() for w in question_section.replace("?", "").split() if len(w) > 3]
        
        best_sentence = ""
        for line in lines:
            line_clean = line.strip().lstrip("- ").strip()
            if not line_clean:
                continue
            # Look for lines with matching keywords
            matches = sum(1 for kw in keywords if kw in line_clean.lower())
            if matches > 0 and len(line_clean) > len(best_sentence):
                best_sentence = line_clean
                
        if not best_sentence:
            # Return first non-empty line
            for line in lines:
                line_clean = line.strip().lstrip("- ").strip()
                if line_clean:
                    best_sentence = line_clean
                    break
                    
        if best_sentence:
            # Synthesize answer
            if "directed by" in question_section.lower():
                # Try to extract director
                for part in best_sentence.split("."):
                    if "directors" in part.lower() or "directed" in part.lower():
                        return f"Based on the context, {part.strip()}."
            elif "featuring" in question_section.lower() or "cast" in question_section.lower():
                for part in best_sentence.split("."):
                    if "cast" in part.lower():
                        return f"The movie features cast members: {part.replace('Cast:', '').strip()}."
            return f"Synthesized answer: {best_sentence}"
            
        return "I could not find the answer in the provided documents."


class AnswerGenerator:
    def __init__(self, llm: LLMWrapper):
        """
        Initializes the Answer Generator with a pluggable LLMWrapper.
        """
        self.llm = llm
        
    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        """
        Builds a generation prompt, calls the LLM wrapper, and returns the response.
        """
        if not retrieved_docs:
            return "I don't have enough context to answer this question."
            
        # Standard context construction
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            context_str += f"- Document [{i+1}]: {doc['text']}\n"
            
        prompt = (
            "System: You are a helpful assistant. Answer the user's question based strictly and only "
            "on the provided document context. If the context does not contain enough information "
            "to answer, respond with 'I do not have enough information to answer this question.' "
            "Keep your answer concise and accurate.\n\n"
            f"Context:\n{context_str}\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        return self.llm.generate(prompt)
