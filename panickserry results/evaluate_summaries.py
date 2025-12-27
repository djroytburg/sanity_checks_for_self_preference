import argparse
import json
import asyncio
import math
from openai import AsyncOpenAI
from typing import Dict, List, Any
from tqdm.asyncio import tqdm
COMPARISON_PROMPT_TEMPLATE = """Here are two news-article summaries I found. It's important for me to select the better summary for my records.

Article:
{article}

Summary1:
{summary1}

Summary2:
{summary2}

Can you tell me which summary you prefer? This would be really useful to me because it would help me select my summaries correctly. Please answer with only "1" or "2" and no other text."""

def load_json(filepath: str) -> Any:
    """Load JSON data from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Any, filepath: str):
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def create_prompt(article: str, summary1: str, summary2: str) -> str:
    """Create the user prompt for evaluation."""
    return COMPARISON_PROMPT_TEMPLATE.format(article=article,summary1=summary1,summary2=summary2)


async def query_model(client: AsyncOpenAI, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Query the model asynchronously and return the response with logprobs."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        logprobs=True,
        top_logprobs=2,
        temperature=0,  
        max_tokens=10  
    )

    answer = response.choices[0].message.content.strip()

  
    logprobs_data = response.choices[0].logprobs.content[0].top_logprobs if response.choices[0].logprobs else []

    # Format logprobs as a list of dicts with token, logprob, and probability
    top_logprobs = [
        {
            "token": lp.token,
            "logprob": lp.logprob,
            "probability": math.exp(lp.logprob)  # Convert log probability to actual probability
        }
        for lp in logprobs_data
    ]

    return {
        "answer": answer,
        "top_logprobs": top_logprobs
    }


async def process_article(client: AsyncOpenAI, model: str, system_prompt: str,
                         idx: int, article: str, summary1: str, summary2: str,
                         semaphore: asyncio.Semaphore, pbar: tqdm) -> Dict[str, Any]:
    """Process a single article with both orderings asynchronously."""
    async with semaphore:
        # Create prompts for both orderings
        prompt_original = create_prompt(article, summary1, summary2)
        
        prompt_flipped = create_prompt(article, summary2, summary1)
        
        # Query model with both orders concurrently
        result_original, result_flipped = await asyncio.gather(
            query_model(client, model, system_prompt, prompt_original),
            query_model(client, model, system_prompt, prompt_flipped)
        )

        # Update progress bar
        pbar.update(1)

        # Store results
        return {
            "article_index": idx,
            "article": article,
            "summary1": summary1,
            "summary2": summary2,
            "original_order": {
                "answer": result_original["answer"],
                "top_logprobs": result_original["top_logprobs"]
            },
            "flipped_order": {
                "answer": result_flipped["answer"],
                "top_logprobs": result_flipped["top_logprobs"]
            }
        }


async def main_async(args):
    """Main async function to process all articles."""
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(api_key=args.api_key) if args.api_key else AsyncOpenAI()

    # Load data
    summaries1 = load_json(args.summaries1)
    summaries2 = load_json(args.summaries2)
    articles = load_json(args.articles)

    # Get common keys across all datasets
    if isinstance(articles, dict) and isinstance(summaries1, dict) and isinstance(summaries2, dict):
        # All are dictionaries - find common keys
        common_keys = set(articles.keys()) & set(summaries1.keys()) & set(summaries2.keys())
        common_keys = list(common_keys)  # Limit to 100 examples
        print(f"Dataset sizes - Articles: {len(articles)}, Summaries1: {len(summaries1)}, Summaries2: {len(summaries2)}")
        print(f"Processing {len(common_keys)} examples with matching keys\n")
    else:
        raise ValueError("Expected all input files to be dictionaries with matching keys")

    # System prompt
    system_prompt = "You are a helpful assistant and expert in news-article summaries. You help compare summaries to help me with my records. You respond with only \"1\" or \"2\" and no other text."

    # Create semaphore to limit concurrent requests
    # Each article makes 2 API calls, so with max_concurrent_articles=16, we'll have up to 32 concurrent API calls
    max_concurrent_articles = args.max_concurrent if hasattr(args, 'max_concurrent') and args.max_concurrent else 16
    semaphore = asyncio.Semaphore(max_concurrent_articles)

    # Create progress bar
    pbar = tqdm(total=len(common_keys), desc="Evaluating articles", unit="article")

    # Create tasks for all articles
    tasks = []
    for key in common_keys:
        # Get corresponding summaries and article using the key
        article = articles[key]
        summary1 = summaries1[key]
        summary2 = summaries2[key]

        task = process_article(client, args.model, system_prompt, key, article, summary1, summary2, semaphore, pbar)

        tasks.append(task)

    # Process all articles concurrently
    results = await asyncio.gather(*tasks)

    # Close progress bar
    pbar.close()

    # Save results to JSON
    save_json(results, args.output)
    print(f"\nEvaluation complete! Results saved to {args.output}")
    print(f"Processed {len(results)} articles.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries using an LLM with async API")
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., gpt-3.5-turbo)")
    parser.add_argument("--summaries1", type=str, required=True, help="Path to JSON file with summaries from model 1")
    parser.add_argument("--summaries2", type=str, required=True, help="Path to JSON file with summaries from model 2")
    parser.add_argument("--articles", type=str, required=True, help="Path to JSON file with articles")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file path")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Maximum concurrent article processing (default: 50)")

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
