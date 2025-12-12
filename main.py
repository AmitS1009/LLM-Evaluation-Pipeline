import argparse
import json
import os
import re
from dotenv import load_dotenv
from src.models import Conversation, Context, Message
from src.pipeline import EvaluationPipeline

load_dotenv()

def load_json_with_comments(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        content = re.sub(r"(?<!:)//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        content = re.sub(r",\s*(?=[\]}])", "", content)

        try:
            return json.loads(content, strict=False)
        except json.JSONDecodeError:
             return json.loads(content, strict=False)

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    parser.add_argument("--conversation_path", type=str, required=True, help="Path to conversation JSON")
    parser.add_argument("--context_path", type=str, required=True, help="Path to context JSON")
    parser.add_argument("--output_path", type=str, default="evaluation_report.json", help="Path to save output JSON")
    parser.add_argument("--model_openai", type=str, default="gpt-3.5-turbo", help="OpenAI model to use (if key present)")
    parser.add_argument("--model_gemini", type=str, default="gemini-pro", help="Gemini model to use (if key present)")

    args = parser.parse_args()

    try:
        conv_data = load_json_with_comments(args.conversation_path)
        ctx_data = load_json_with_comments(args.context_path)

        conversation = Conversation.model_validate(conv_data)
        context = Context.model_validate(ctx_data)

        # Pass model config to pipeline
        pipeline = EvaluationPipeline(model_openai=args.model_openai, model_gemini=args.model_gemini)
        result = pipeline.run(conversation, context)

        print(json.dumps(result.dict(), indent=2))
        
        with open(args.output_path, 'w') as f:
            json.dump(result.dict(), f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
