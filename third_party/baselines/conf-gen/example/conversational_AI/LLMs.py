import os
from dotenv import load_dotenv
from openai import OpenAI
import yaml
import re
import json
from utils_conversational import SPECIAL_SEP_TOKEN

# Set up the OpenAI API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load LLM prompts
def load_prompts() -> dict:
    prompt_filename = 'llm_prompts.yaml'
    with open(prompt_filename, 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts


class LLM_Consolidate():
    def __init__(self, model="gpt-4o"):
        self.model = model
        print(f"Initialized consolidate LLM using {self.model}!")

    def _format_multiround_questions(self, question: str) -> str:
        sentences = question.strip().split(SPECIAL_SEP_TOKEN)

        formatted_multiround_question = []
        roles = ["User", "Assistant"]

        for i, sentence in enumerate(sentences):
            role = roles[i%2]
            formatted_multiround_question.append(f"{role}: {sentence}")

        return "\n".join(formatted_multiround_question)
    
    def consolidate_question(self, question_raw: str) -> str:
        system_prompt = load_prompts()["consolidation_prompt"]

        fewshot_example = load_prompts()["consolidation_examples"]

        question_raw_formatted = self._format_multiround_questions(question_raw) \
                                    if SPECIAL_SEP_TOKEN in question_raw \
                                    else question_raw

        user_prompt = f"""
                        Passage:
                            {question_raw_formatted}
                        Return only the consolidated question.
                       """

        response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                                    {
                                        "role": "system",
                                        "content": system_prompt
                                    },
                                    {
                                        "role": "system",
                                        "content": fewshot_example
                                    },
                                    {
                                        "role": "user",
                                        "content": user_prompt
                                    }
                                 ],
                        temperature = 0.1,         
                        max_tokens = 64)

        consolidated_question = response.choices[0].message.content.strip()
        # safety trim: keep single line, strip trailing punctuation spaces
        consolidated_question = " ".join(consolidated_question.split())
        
        return consolidated_question
    

class LLM_Respond():
    def __init__(self, model: str="gpt-4o"):
        self.model = model
        print(f"Initialized respond LLM using {self.model}!")

    
    def respond(self, consolidated_question: str) -> str:
        system_prompt = load_prompts()["generation_prompt"]

        fewshot_example = load_prompts()["generation_examples"]

        response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                                    {
                                        "role": "system",
                                        "content": system_prompt
                                    },
                                    {
                                        "role": "system",
                                        "content": fewshot_example
                                    },
                                    {
                                        "role": "user",
                                        "content": consolidated_question
                                    }
                                 ],
                        max_tokens = 150)

        answer = response.choices[0].message.content
        
        return answer


class LLM_Score():
    def __init__(self, model: str="gpt-4o"):
        self.model = model
        print(f"Initialized scoring LLM using {self.model}!")
        
    
    def score(self, 
              consolidated_question: str, 
              response: str,
              verbose: bool) -> float:
        system_prompt = load_prompts()["scoring_prompt"]

        fewshot_example = load_prompts()["scoring_examples"]
        
        user_prompt = f"""
                        Evaluate clarity and policy compliance strictly.
                        Here is the question-answer pair:
                            "question": "{consolidated_question}" 
                            "answer": "{response}"                        
                       """

        tools=[{
            "type": "function",
            "function": {
                "name": "return_score",
                "description": "Return score and rationale.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"]
                }
            }
        }]

        response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                                    {
                                        "role": "system",
                                        "content": system_prompt
                                    },
                                    {
                                        "role": "system",
                                        "content": fewshot_example
                                    },
                                    {
                                        "role": "user",
                                        "content": user_prompt
                                    }
                                 ],
                        tools=tools,
                        tool_choice={"type": "function", 
                                      "function": {"name": "return_score"}},
                        max_tokens = 150,
                        temperature = 0,
                        top_p = 1,
                        n = 1)

        answer_raw = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

        # Extract values
        score = float(answer_raw["score"])
        rationale = answer_raw["rationale"]

        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            raise ValueError(f"Invalid score value returned: {score}")
        
        if verbose:
            return score, rationale
        return score
    