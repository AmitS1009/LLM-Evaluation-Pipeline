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
        # 1. Remove // comments but protect URLs (https://)
        content = re.sub(r"(?<!:)//.*", "", content)
        # 2. Remove /* */ comments (multiline)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        
        # 3. Remove trailing commas (e.g. [1, 2,] -> [1, 2])
        # Match matches comma, whitespace, then ] or }
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
    
    args = parser.parse_args()

    try:
        conv_data = load_json_with_comments(args.conversation_path)
        ctx_data = load_json_with_comments(args.context_path)

        # Parse using Pydantic models
        conversation = Conversation.model_validate(conv_data)
        context = Context.model_validate(ctx_data)

        # Run pipeline
        pipeline = EvaluationPipeline()
        result = pipeline.run(conversation, context)

        # Print to console
        print(json.dumps(result.dict(), indent=2))
        
        # Save to file
        with open(args.output_path, 'w') as f:
            json.dump(result.dict(), f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        # print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
