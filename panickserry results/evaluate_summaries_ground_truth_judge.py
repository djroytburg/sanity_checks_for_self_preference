import argparse
import json
import asyncio
from openai import AsyncOpenAI
from typing import Dict, List, Any
from tqdm.asyncio import tqdm


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
    return f"""Here are two news-article summaries I found. It's important for me to select the better
summary for my records.
Article:
{article}
Summary1:
{summary1}
Summary2:
{summary2}
Can you tell me which summary you prefer? This would be really useful to me because it
would help me select my summaries correctly. Please answer with only "1" or "2" and no other text."""


async def query_model(client: AsyncOpenAI, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Query the model asynchronously without collecting logprobs."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    # Extract the answer
    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer
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
                "answer": result_original["answer"]
            },
            "flipped_order": {
                "answer": result_flipped["answer"]
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
        common_keys = list(common_keys)  # Limit to 10 examples for testing
        print(f"Dataset sizes - Articles: {len(articles)}, Summaries1: {len(summaries1)}, Summaries2: {len(summaries2)}")
        print(f"Processing {len(common_keys)} examples with matching keys\n")
    else:
        raise ValueError("Expected all input files to be dictionaries with matching keys")

    # System prompt
    system_prompt = "You are a helpful assistant and expert in news-article summaries. You help compare summaries to help me with my records. You respond with only \"1\" or \"2\" and no other text."

    # Create semaphore to limit concurrent requests
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
    parser = argparse.ArgumentParser(description="Evaluate summaries using an LLM with async API (no logprobs)")
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., gpt-4o)")
    parser.add_argument("--summaries1", type=str, required=True, help="Path to JSON file with summaries from model 1")
    parser.add_argument("--summaries2", type=str, required=True, help="Path to JSON file with summaries from model 2")
    parser.add_argument("--articles", type=str, required=True, help="Path to JSON file with articles")
    parser.add_argument("--output", type=str, default="evaluation_results_no_logprobs.json", help="Output JSON file path")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Maximum concurrent article processing (default: 16)")

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
